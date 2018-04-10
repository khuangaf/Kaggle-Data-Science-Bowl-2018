import os, sys
sys.path.append(os.path.dirname(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from dataset.reader import *
#--------------------------------------------------------------
from skimage.exposure import adjust_gamma
from train_resnext import * #se_resnext50_mask_rcnn_2crop
#from train_mask_rcnn_net_2 import * #se_resnext101_mask_rcnn


#--------------------------------------------------------------
from predict_arguments import *

## color argument
def unsharp(image):
    #blur = cv2.GaussianBlur(image, (9,9), 10.0)
    blur = cv2.GaussianBlur(image, (5,5), 5.0)
    image = cv2.addWeighted(image, 1.5, blur, -0.5, 0, image)
    return image


def clahe(image,clip=2,grid=16):

    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid)) #2,8
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = c.apply(l_channel)

    lab   = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def run_predict():

    # out_dir = RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-01'
    # initial_checkpoint = \
    #    RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-01/checkpoint/00011000_model.pth'

    # out_dir = RESULTS_DIR + '/mask-se-resnext101-gray500-border0.25-02'
    # initial_checkpoint = \
    #    RESULTS_DIR + '/mask-se-resnext101-gray500-border0.25-02/checkpoint/00036000_model.pth'

#     out_dir = RESULTS_DIR + '/mask-rcnn-50-resnext-gray500-aug-avg'
    out_dir = RESULTS_DIR + '/mask-rcnn-resnext-50-color_external130-aug-avg'
    initial_checkpoint = \
       RESULTS_DIR + '/mask-rcnn-resnext-50-color_external130/checkpoint/00020500_model.pth'
       #RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-05b/checkpoint/00015500_model.pth'
       #RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-05a/checkpoint/00014500_model.pth'
        #'/root/share/project/kaggle/science2018/results/mask-se-resnext50-gray500-00/checkpoint/00034000_model.pth'


    # augment -----------------------------------------------------------------------------------------------------
#     do_test_augment, undo_test_augment = do_test_augment_identity, undo_test_augment_identity
#     do_test_augment, undo_test_augment = do_test_augment_horizontal_flip, undo_test_augment_horizontal_flip
#     do_test_augment, undo_test_augment = do_test_augment_vertical_flip, undo_test_augment_vertical_flip

#     do_test_augment, undo_test_augment = do_test_augment_rotate180, undo_test_augment_rotate180
    do_test_augment, undo_test_augment = do_test_augment_rotate090, undo_test_augment_rotate090
#     do_test_augment, undo_test_augment = do_test_augment_rotate270, undo_test_augment_rotate270

#     do_test_augment, undo_test_augment = do_test_augment_scale, undo_test_augment_scale
    # augment -----------------------------------------------------------------------------------------------------



    # split = 'valid1_ids_gray2_43'
#     split = 'test1_ids_gray_only_53'
    
    split = 'test1_ids_color_12'
    # split = 'BBBC006'

    #tag = 'test1_ids_gray2_53-00011000_model'
    #tag = 'xxx_scale_2.4_high'
#     tag = 'identity'
#     tag = 'h_flip'
#     tag = 'v_flip'
#     tag = 'r_180'
    tag = 'r_90'

    

    ## setup  --------------------------
    os.makedirs(out_dir +'/predict/%s/overlays'%tag, exist_ok=True)
    os.makedirs(out_dir +'/predict/%s/predicts'%tag, exist_ok=True)
    os.makedirs(out_dir +'/predict/%s/npys'%tag, exist_ok=True)

    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    
    log.write('\n')


    ## net ------------------------------
    cfg = Configuration()
    cfg.rcnn_test_nms_pre_score_threshold = 0.5
    cfg.mask_test_nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold

    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')
    log.write('\ttsplit   = %s\n'%(split))
    log.write('\tlen(ids) = %d\n'%(len(ids)))
#     log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('tag=%s\n'%tag)
    log.write('\n')




    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')

    for i in range(len(ids)):
        folder, name = ids[i].split('/')[-2:]
        print('%03d %s'%(i,name))

        #'4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2'
        #name='0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6'
        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)


        ###--------------------------------------
        augment_image = do_test_augment(image)

        #augment_image = cv2.blur(augment_image,(15,15))
        #augment_image = unsharp(augment_image)
        #augment_image = adjust_gamma(augment_image,0.5)
        #augment_image = clahe(augment_image, clip=2, grid=16)

        net.set_mode('test')
        with torch.no_grad():
            input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
            input = Variable(input).cuda()
            net.forward(input)


        rcnn_proposal, detection, mask, mask_score  = undo_test_augment(net, image)



        ##save results ---------------------------------------
        np.save(out_dir +'/predict/%s/npys/%s.npy'%(tag,name),mask)

        #----
        if 1:
            norm_image = adjust_gamma(image,2.5)

            threshold = 0.8  #cfg.rcnn_test_nms_pre_score_threshold  #0.8
            #all1 = draw_predict_proposal(threshold, image, rcnn_proposal)
#             all2 = draw_predict_mask(threshold, image, mask, detection)

            ## save
            #cv2.imwrite(out_dir +'/predict/%s/predicts/%s.png'%(tag,name), all1)
#             cv2.imwrite(out_dir +'/predict/%s/predicts/%s.png'%(tag,name), all2)

            #image_show('predict_proposal',all1)
#             image_show('predict_mask',all2)

            if 1:
                norm_image      = adjust_gamma(image,2.5)
                color_overlay   = multi_mask_to_color_overlay(mask)
                color1_overlay  = multi_mask_to_contour_overlay(mask, color_overlay)
                contour_overlay = multi_mask_to_contour_overlay(mask, norm_image, [0,255,0])


                #mask_score = cv2.cvtColor((np.clip(mask_score,0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                mask_score = cv2.cvtColor((mask_score/mask_score.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

                all = np.hstack((image, contour_overlay, color1_overlay, mask_score)).astype(np.uint8)
#                 image_show('overlays',all)

                #psd
                os.makedirs(out_dir +'/predict/overlays', exist_ok=True)
                cv2.imwrite(out_dir +'/predict/%s/overlays/%s.png'%(tag,name),all)

                os.makedirs(out_dir +'/predict/%s/overlays/%s'%(tag,name), exist_ok=True)
                cv2.imwrite(out_dir +'/predict/%s/overlays/%s/%s.png'%(tag,name,name),image)
                cv2.imwrite(out_dir +'/predict/%s/overlays/%s/%s.mask.png'%(tag,name,name),color_overlay)
                cv2.imwrite(out_dir +'/predict/%s/overlays/%s/%s.contour.png'%(tag,name,name),contour_overlay)

            cv2.waitKey(0)




    #assert(test_num == len(test_loader.sampler))
    log.write('-------------\n')
    
    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_predict()

    print('\nsucess!')