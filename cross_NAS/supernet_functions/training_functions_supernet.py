import torch
from torch.autograd import Variable
import time
import numpy as np
from general_functions.utils import save, ap_per_class, xywh2xyxy, non_max_suppression, get_batch_statistics

from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.model_supernet import MixedOperation
from general_functions.prune_utils import BNOptimizer, get_sr_flag
from general_functions.quan import QuanConv2d, QuanAct, quantizer
from building_blocks.layers import BatchNorm2d
from building_blocks.builder import ConvBNRelu


def model_quantize(model):
    for m in model.modules():
        if isinstance(m, ConvBNRelu):
            m[0] = QuanConv2d(m[0],
                quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']),
                quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
            if not isinstance(m[-1], BatchNorm2d):
                m[-1] = QuanAct(m[-1], quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))

                
class TrainerSupernet:
    def __init__(self, w_criterion, theta_criterion, w_optimizer, theta_optimizer, w_scheduler, 
                 scale_sparse_rate, start_quant, train_thetas_from_the_epoch, logger, writer):
        self.logger = logger
        self.writer = writer

        self.w_criterion = w_criterion
        self.theta_criterion = theta_criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler

        self.cnt_epochs = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.start_quant = start_quant
        self.train_thetas_from_the_epoch = train_thetas_from_the_epoch
        self.print_freq = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model = CONFIG_SUPERNET['train_settings']['path_to_save_model']
        self.path_to_save_current_model = CONFIG_SUPERNET['train_settings']['path_to_save_current_model']

        self.scale_sparse_rate = scale_sparse_rate
        self.sr = CONFIG_SUPERNET['prune']['sr']
        
        self._valid_iter = None

    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):

        best_mAP = 0.0

        # start to train without quantization
#         for epoch in range(self.start_quant):
#             self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
#             self.logger.info("Firstly, start to train weights for epoch %d" % epoch)
            
#             self.weight_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
#             self.w_scheduler.step()
        
#         # quantize the model
#         print("quantize the model")
#         model_quantize(model)
#         model.cuda()
        
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights without quantization for epoch %d" % epoch)
            
            self.weight_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
           
        # then, train weights and theta sequentially
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights and thetas for epoch %d" % epoch)
            self.train_w_and_thetas(model, train_w_loader, train_thetas_loader, self.w_optimizer,                                               self.theta_optimizer, epoch)
            self.w_scheduler.step()
            mAP = self._validate(model, test_loader, epoch, img_size=CONFIG_SUPERNET['dataloading']['img_size'])

            state = {
                "epoch" : epoch + 1,
                "state_dict" : model.state_dict(),
                "w_optimizer" : self.w_optimizer.state_dict(),
                "theta_optimizer" : self.theta_optimizer.state_dict(),
                "w_scheduler" : self.w_scheduler
            }
            save(state, self.path_to_save_current_model) # save the supernet

            if mAP is not None:
                self.logger.info("current mAP is : %f" % mAP)
                if best_mAP < mAP:
                    best_mAP = mAP
                    self.logger.info("Best mAP by now. Save model")
                    save(state, self.path_to_save_model)

    def weight_step(self, model, loader, optimizer, epoch, info_for_logger=''):
        """
        used for updating weight param.
        """
        model.train()

        sr_flag = get_sr_flag(epoch, self.sr)

        for step, (_, images, targets) in enumerate(loader):
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            model.reset_binary_gates() # random sample binary gates
            model.unused_modules_off() # remove unused module for speedup

            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            outs, latency_to_accumulate = model(images, latency_to_accumulate)
            
            # if step > 1:
            #     print(model.module_list[0][0].weight.grad.data[0])
            # print(outs[0][0][0])
            loss, ce, lat, loss_components = self.w_criterion(outs, targets, latency_to_accumulate, model)

            optimizer.zero_grad()
            ce.backward()

            BNOptimizer.updateBN(sr_flag, model, self.scale_sparse_rate)

            optimizer.step() # update weight parameters

            # unused modules back
            model.unused_modules_back()

            self._train_logging(loss, ce, lat, loss_components, step, epoch, model,
                                len_loader=len(loader),
                                info_for_logger=info_for_logger)

    def gradient_step(self, model, loader, optimizer, epoch, step, info_for_logger=''):
        """
        used for updating arch. param.
        """
        model.train()
        # Mix edge mode
        MixedOperation.MODE = CONFIG_SUPERNET['binary_mode']
        
        if self._valid_iter is None:
            self._valid_iter = iter(loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(loader)
            data = next(self._valid_iter)
            
        _, images, targets = data
        
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        # compute output
        model.reset_binary_gates()  # random sample binary gates
        model.unused_modules_off()  # remove unused module for speedup

        latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
        outs, latency_to_accumulate = model(images, latency_to_accumulate)
        loss, ce, lat, loss_components = self.theta_criterion(outs, targets, latency_to_accumulate, model)

        # compute gradient and do SGD step
        optimizer.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        model.set_arch_param_grad()
        optimizer.step()
        if MixedOperation.MODE == 'two_v2':
            model.rescale_updated_arch_param()
        # back to normal mode
        model.unused_modules_back()

        self._train_logging(loss, ce, lat, loss_components, step, epoch, model,
                            len_loader=len(loader),
                            info_for_logger="_thetas")

        MixedOperation.MODE = None
    
    def train_w_and_thetas(self, model, train_loader, valid_loader, train_optimizer, valid_optimizer,
                           epoch):
        model.train()

        sr_flag = self.sr

        for step, (_, images, targets) in enumerate(train_loader):
            # self.logger.info("Start to train weights")
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            model.reset_binary_gates()  # random sample binary gates
            model.unused_modules_off()  # remove unused module for speedup

            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            outs, latency_to_accumulate = model(images, latency_to_accumulate)

            loss, ce, lat, loss_components = self.w_criterion(outs, targets, latency_to_accumulate, model)

            train_optimizer.zero_grad()
            ce.backward()

            BNOptimizer.updateBN(sr_flag, model, self.scale_sparse_rate)

            train_optimizer.step()  # update weight parameters

            # unused modules back
            model.unused_modules_back()

            self._train_logging(loss, ce, lat, loss_components, step, epoch, model,
                                len_loader=len(train_loader),
                                info_for_logger="_weights")
            
            # self.logger.info("Start to train thetas")
            self.gradient_step(model, valid_loader, valid_optimizer, epoch, step, info_for_logger='')
    
    def _validate(self, model, loader, epoch, img_size):
        model.eval()
        start_time = time.time()
        labels = []
        sample_metrics = []
        
        with torch.no_grad():
            for step, (_, images, targets) in enumerate(loader):
                images, targets = images.cuda(), targets.cuda()
                # Extract labels
                labels += targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= img_size

                images = Variable(images, requires_grad=False)

                # set chosen op active
                model.set_chosen_op_active()
                # remove unused modules
                model.unused_modules_off()

                latency_to_accumulate = torch.Tensor([[0.0]]).cuda()

                outs, latency_to_accumulate = model(images, latency_to_accumulate)
                outs = non_max_suppression(outs, conf_thres=CONFIG_SUPERNET['valid_settings']['conf_thres'],
                                           iou_thres=CONFIG_SUPERNET['valid_settings']['nms_thres'])

                sample_metrics += get_batch_statistics(outs, targets.cpu(),
                                                       iou_threshold=CONFIG_SUPERNET['valid_settings']['iou_thres'])

                # unused modules back
                model.unused_modules_back()

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            self._valid_logging(start_time=start_time, epoch=epoch, metrics_output=metrics_output)
        else:
            self.logger.info(
                " mAP not measured (no detections found by model) for {:3d}/{} Time {:.2f}".format(
                                             epoch + 1, self.cnt_epochs, time.time() - start_time))
            return
        return AP.mean()

    def _valid_logging(self, start_time, epoch, metrics_output):
        precision, recall, AP, f1, ap_class = metrics_output

        # self.writer.add_scalar('valid_precision', precision.mean().item(), epoch)
        # self.writer.add_scalar('valid_recall', recall.mean().item(), epoch)
        self.writer.add_scalar('valid_mAP', AP.mean().item(), epoch)
        # self.writer.add_scalar('valid_f1', f1.mean().item(), epoch)

        self.logger.info("valid : [{:3d}/{}] Final Precision {:.4%}, Time {:.2f}".format(
            epoch + 1, self.cnt_epochs, AP.mean().item(), time.time() - start_time))

    def _train_logging(self, loss, ce, lat, loss_components, step, epoch, model,
                                    len_loader, info_for_logger=''):

        # self.writer.add_scalar('total_loss', loss.item(), epoch)
        self.writer.add_scalar('ce_loss', ce.item(), epoch)
        self.writer.add_scalar('latency_loss', lat.item(), epoch)
        # self.writer.add_scalar('iou_loss', loss_components[0].item(), epoch)
        # self.writer.add_scalar('obj_loss', loss_components[1].item(), epoch)
        # self.writer.add_scalar('cls_loss', loss_components[2].item(), epoch)
        if epoch % 2 == 0:
            probs = model.module_list[2].probs_over_ops
            self.writer.add_scalar('alpha_k3', probs[0].item(), epoch)
            self.writer.add_scalar('alpha_k5', probs[1].item(), epoch)
            self.writer.add_scalar('alpha_k7', probs[2].item(), epoch)
            self.writer.add_scalar('alpha_skip', probs[3].item(), epoch)
        
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info("training" + info_for_logger +
                             ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
                             "ce_loss {:.3f}, lat_loss {:.3f} "
                             "iou_loss {:.3f}, obj_loss {:.3f}, cls_loss {:.3f}".format(
                                 epoch + 1, self.cnt_epochs, step, len_loader - 1, loss.item(), ce.item(), lat.item(),
                                 loss_components[0].item(), loss_components[1].item(), loss_components[2].item()))
