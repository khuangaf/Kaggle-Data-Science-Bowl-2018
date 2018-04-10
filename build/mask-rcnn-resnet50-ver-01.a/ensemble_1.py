import os, sys
sys.path.append(os.path.dirname(__file__))

from common import *
from dataset.reader import *
from skimage.exposure import adjust_gamma
from submit_resnext import *
#ensemble =======================================================
ALL_TEST_IMAGE_ID =[
    '0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5',
    '0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac',
    '0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732',
    '0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925',
    '0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6',
    '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466',
    '1747f62148a919c8feb6d607faeebdf504b5e2ad42b6b1710b1189c37ebcdb2c',
    '17b9bf4356db24967c4677b8376ac38f826de73a88b93a8d73a8b452e399cdff',
    '1879f4f4f05e2bada0ffeb46c128b8df7a79b14c84f38c3e216a69653495153b',
    '191b2b2205f2f5cc9da04702c5d422bc249faf8bca1107af792da63cccfba829',
    '1962d0c5faf3e85cda80e0578e0cb7aca50826d781620e5c1c4cc586bc69f81a',
    '1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e',
    '1d9eacb3161f1e2b45550389ecf7c535c7199c6b44b1c6a46303f7b965e508f1',
    '1ef68e93964c2d9230100c1347c328f6385a7bc027879dc3d4c055e6fe80cb3c',
    '259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
    '295682d9eb5acb5c1976a460c085734bfaf38482b0a3f02591c2bfdcd4128549',
    '31f1fbe85b8899258ea5bcf5f93f7ac8238660c386aeab40649c715bd2e38a0a',
    '336d3e4105766f8ad328a7ee9571e743f376f8cbcf6a969ca7e353fe3235c523',
    '38f5cfb55fc8b048e82a5c895b25fefae7a70c71ab9990c535d1030637bf6a1f',
    '3c4c675825f7509877bc10497f498c9a2e3433bf922bd870914a2eb21a54fd26',
    '432f367a4c5b5674de2e2977744d10289a064e5704b21af6607b4975be47c580',
    '43a71aeb641faa18742cb826772a8566c6c947d7050f9ab15459de6cc2b3b6af',
    '44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078',
    '4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2',
    '472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',
    '4be73d68f433869188fe5e7f09c7f681ed51003da6aa5d19ce368726d8e271ee',
    '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac',
    '505bc0a3928d8aef5ce441c5a611fdd32e1e8eccdc15cc3a52b88030acb50f81',
    '519dc0d672d1c295fc69b629af8721ccb1a1f136d1976685a68487e62547ffe0',
    '51c70bb8a299943b27f8b354571272692d8f2705036a1a9562156c76da5f025b',
    '52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde',
    '53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4',
    '550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18',
    '5cee644e5ffbef1ba021c7f389b33bafd3b1841f04d3edd7922d5084c2c4e0c7',
    '648c8ffa496e1716017906d0bf135debfc93386ae86aa3d4adbda9a505985fd9',
    '697a05c6fe4a07c601d46da80885645ad574ea19b47ee795ccff216c9f1f1808',
    '699f2992cd71e2e28cf45f81347ff22e76b37541ce88087742884cd0e9aadc68',
    '78a981bd27ba0c65a9169548665a17bda9f49050d0d3893a6567d1eb92cd003d',
    '7bdb668e6127b7eafc837a883f0648002bd063c736f55a4f673e787250a3fb04',
    '7f4cbe0b36b5d09466476a7d4e01f4f976c67872d549f4ff47b3e1e3a2b403af',
    '8922a6ac8fd0258ec27738ca101867169b20d90a60fc84f93df77acd5bf7c80b',
    '8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab',
    '912a679e4b9b1d1a75170254fd675b8c24b664d80ad7ea7e460241a23535a406',
    '9ab2d381f90b485a68b82bc07f94397a0373e3215ad20935a958738e55f3cfc2',
    '9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab',
    'a4816cc1fb76cb3c5e481186833fc0ae9cf426a1406a2607e974e65e9cddba4f',
    'a984e7fb886aa02e29d112766d3ce26a4f78eac540ce7bbdbd42af2761928f6d',
    'ab298b962a63e4be9582513aaa84a5e270adba5fd2b16a50e59540524f63c3b8',
    'ade080c6618cbbb0a25680cf847f312b5e19b22bfe1cafec0436987ebe5b1e7e',
    'b83d1d77935b6cfd44105b54600ffc4b6bd82de57dec65571bcb117fa8398ba3',
    'bdc789019cee8ddfae20d5f769299993b4b330b2d38d1218646cf89e77fbbd4d',
    'c8e79ff4ac55f4b772057de28e539727b7f4f2a3de73bf7a082a0ace86d609eb',
    'ca20076870e8fb604e61802605a9ac45419c82dd3e23404c56c4869f9502a5ef',
    'd616d323a9eeb9da1b66f8d5df671d63c092c9919cb2c0b223e29c63257c944d',
    'd6eb7ce7723e2f6dc13b90b41a29ded27dbd815bad633fdf582447c686018896',
    'd8d4bf68a76e4e4c5f21de7ac613451f7115a04db686151e78b8ec0b6a22022b',
    'da6c593410340b19bb212b9f6d274f95b08c0fc8f2570cd66bc5ed42c560acab',
    'dab46d798d29aff2e99c23f47ed3064f5cafb1644629b015c95a2dd2ee593bb4',
    'df40099c6306ca1f47fcc8a62e2fa39486d4e223177afdc51b2ad189691802d8',
    'e17b7aedd251a016c01ef9158e6e4aa940d9f1b35942d86028dc1222192a9258',
    'eea70a7948d25a9a791dbcb39228af4ea4049fe5ebdee9c04884be8cca3da835',
    'f0d0ab13ff53adc3c4d57e95a5f83d80b06f2cbc0bf002b52cf7b496612e0ce4',
    'f5effed21f671bbf4551ecebb7fe95f3be1cf09c16a60afe64d2f0b95be9d1eb',
    'fac507fa4d1649e8b24c195d990f1fc3ca3633d917839e1751a9d412a14ab5e3',
    'fe9adb627a6f45747c5a8223b671774791ededf9364f6544be487c540107fa4f',
]
class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members=[]
        self.center =[]

    def add_item(self, box, score, instance):
        if self.center ==[]:
            self.members = [{
                'box': box, 'score': score, 'instance': instance
            },]
            self.center  = {
                'box': box, 'score': score, 'union':(instance>0.5), 'inter':(instance>0.5),
            }
        else:
            self.members.append({
                'box': box, 'score': score, 'instance': instance
            })
            center_box   = self.center['box'].copy()
            center_score = self.center['score']
            center_union = self.center['union'].copy()
            center_inter = self.center['inter'].copy()

            self.center['box'] = [
                min(box[0],center_box[0]),
                min(box[1],center_box[1]),
                max(box[2],center_box[2]),
                max(box[3],center_box[3]),
            ]
            self.center['score'] = max(score,center_score)
            self.center['union'] = center_union | (instance>0.5)
            self.center['inter'] = center_inter & (instance>0.5)

    def distance(self, box, score, instance):
        center_box   = self.center['box']
        center_union = self.center['union']
        center_inter = self.center['inter']

        x0 = int(max(box[0],center_box[0]))
        y0 = int(max(box[1],center_box[1]))
        x1 = int(min(box[2],center_box[2]))
        y1 = int(min(box[3],center_box[3]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        if box_intersection<0.01: return 0

        x0 = int(min(box[0],center_box[0]))
        y0 = int(min(box[1],center_box[1]))
        x1 = int(max(box[2],center_box[2]))
        y1 = int(max(box[3],center_box[3]))

        i0 = center_union[y0:y1,x0:x1]  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.logical_and(i0, i1).sum()
        area = np.logical_or(i0, i1).sum()
        overlap = intersection/(area + 1e-12)

        return overlap




def do_clustering( boxes, scores, instances, threshold=0.5):

    clusters = []
    num_arguments   = len(instances)
    for n in range(0,num_arguments):
        box   = boxes[n]
        score = scores[n]
        instance = instances[n]

        num = len(instance)
        for m in range(num):
            b, s, i = box[m],score[m],instance[m]

            is_group = 0
            for c in clusters:
                iou = c.distance(b, s, i)

                if iou>threshold:
                    c.add_item(b, s, i)
                    is_group=1

            if is_group == 0:
                c = Cluster()
                c.add_item(b, s, i)
                clusters.append(c)

    return clusters


def mask_to_more(mask):
    H,W      = mask.shape[:2]
    box      = []
    score    = []
    instance = []

#     for i in range(mask.max()):
#     print(set(mask.flatten()))
    for i in range(int(mask.max())):
        m = (mask==(i+1))

        #filter by size, boundary, etc ....
        if 1:

            #box
            y,x = np.where(m)
            try:
#             print(y)
                y0 = y.min()
                y1 = y.max()
                x0 = x.min()
                x1 = x.max()
            except ValueError:
                continue
            
            b = [x0,y0,x1,y1]

            #score
            s = 1

            # add --------------------
            box.append(b)
            score.append(s)
            instance.append(m)

            # image_show('m',m*255)
            # cv2.waitKey(0)

    box      = np.array(box,np.float32)
    score    = np.array(score,np.float32)
    instance = np.array(instance,np.float32)

    if len(box)==0:
        box      = np.zeros((0,4),np.float32)
        score    = np.zeros((0,1),np.float32)
        instance = np.zeros((0,H,W),np.float32)

    return box, score, instance
out_dir = \
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/out'

def run_ensemble():

    
        #'/root/share/project/kaggle/science2018/results/__ensemble__/xxx'

    ensemble_dirs = [
        #different predictors, test augments, etc ...

        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/evaluate_test/test1_ids_gray2_53-00011000_model',
        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/evaluate_test/test1_ids_gray2_53-00017000_model',
        # '/root/share/project/kaggle/science2018/results/__submit__/LB-0.523/npys-0.570'

        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx',
        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_horizontal_flip',
        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_vertical_flip',
        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_scale_1.2',
        # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_scale_0.8',
#         '/data/steeve/DSB/results/mask-rcnn-50-resnext-gray500-aug/predict/h_flip',
#         '/data/steeve/DSB/results/mask-rcnn-50-resnext-gray500-aug/predict/v_flip',
#         '/data/steeve/DSB/results/mask-rcnn-50-resnext-gray500-aug/predict/r_180',
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/identity',
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/h_flip',
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/v_flip',
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/r_180',
        '/data/steeve/DSB/results/mask-rcnn-resnext-50-color_external130-aug-avg/predict/r_90',        
     
#         '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/original',
#         '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/horizontal_flip',
#         '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/vertical_flip',
#         '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/scale_1.2',
#         '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/scale_0.8',
    ]

    ## setup  --------------------------
    os.makedirs(out_dir +'/average_semantic_mask', exist_ok=True)
    os.makedirs(out_dir +'/cluster_union_mask', exist_ok=True)
    os.makedirs(out_dir +'/cluster_inter_mask', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_mask', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_mask_overlays', exist_ok=True)
    os.makedirs(out_dir +'/submit/npys', exist_ok=True)
    
    names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    names = [n.split('/')[-2]for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for name in names:
        #name='1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e'
        print(name)
        boxes=[]
        scores=[]
        instances=[]

        average_semantic_mask = None
        for dir in ensemble_dirs:
            npy_file = dir +'/npys/%s.npy'%name
            mask = np.load(npy_file)
#             png_file   = dir +'/overlays/%s/%s.mask.png'%(name,name)
#             mask_image = cv2.imread(png_file,cv2.IMREAD_COLOR)
#             mask       = image_to_mask(mask_image)

            if average_semantic_mask is None:
                average_semantic_mask = (mask>0).astype(np.float32)
            else:
                average_semantic_mask = average_semantic_mask + (mask>0).astype(np.float32)

            # color_overlay = mask_to_color_overlay(mask)
            # image_show('color_overlay',color_overlay)
            # image_show('average_semantic_mask',average_semantic_mask*255)
            # cv2.waitKey(0)

            box, score, instance = mask_to_more(mask)
            boxes.append(box)
            scores.append(score)
            instances.append(instance)

        clusters = do_clustering( boxes, scores, instances, threshold=0.3)
        H,W      = average_semantic_mask.shape[:2]


        # <todo> do your ensemble  here! =======================================
        ensemble_mask = np.zeros((H,W), np.int32)
        for i,c in enumerate(clusters):
            num_members = len(c.members)
            average = np.zeros((H,W), np.float32)  #e.g. use average
            for n in range(num_members):
                average = average + c.members[n]['instance']
            average = average/num_members

            ensemble_mask[average>0.5] = i+1

        #do some post processing here ---
        # e.g. fill holes
        #      remove small fragment
        #      remove boundary
        # <todo> do your ensemble  here! =======================================




        # show clustering/ensmeble results
        cluster_inter_mask = np.zeros((H,W), np.int32)
        cluster_union_mask = np.zeros((H,W), np.int32)
        for i,c in enumerate(clusters):
            cluster_inter_mask[c.center['inter']]=i+1
            cluster_union_mask[c.center['union']]=i+1

            # image_show('all',all/num_members*255)
            # cv2.waitKey(0)
            # pass

        color_overlay0 = multi_mask_to_color_overlay(cluster_inter_mask)
        color_overlay1 = multi_mask_to_color_overlay(cluster_union_mask)
        color_overlay2 = multi_mask_to_color_overlay(ensemble_mask)
        ##-------------------------
        average_semantic_mask = (average_semantic_mask/num_ensemble*255).astype(np.uint8)
        average_semantic_mask = cv2.cvtColor(average_semantic_mask,cv2.COLOR_GRAY2BGR)

        cv2.imwrite(out_dir +'/average_semantic_mask/%s.png'%(name),average_semantic_mask)
        cv2.imwrite(out_dir +'/cluster_inter_mask/%s.mask.png'%(name),color_overlay0)
        cv2.imwrite(out_dir +'/cluster_union_mask/%s.mask.png'%(name),color_overlay1)
        cv2.imwrite(out_dir +'/ensemble_mask/%s.mask.png'%(name),color_overlay2)


#         image_show('average_semantic_mask',average_semantic_mask)
#         image_show('cluster_inter_mask',color_overlay0)
#         image_show('cluster_union_mask',color_overlay1)
        #image_show('ensemble_mask',color_overlay2)

        if 1:
            folder = 'stage1_test'
            image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)

            mask = ensemble_mask
            norm_image      = adjust_gamma(image,2.5)
            color_overlay   = multi_mask_to_color_overlay(mask)
            color1_overlay  = multi_mask_to_contour_overlay(mask, color_overlay)
            contour_overlay = multi_mask_to_contour_overlay(mask, norm_image, [0,255,0])
            all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
#             image_show('ensemble_mask',all)

            #psd
#             print(mask.shape)
            np.save(out_dir +'/submit/npys/%s.npy'%(name),mask)
        
            cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s.png'%(name),all)
            os.makedirs(out_dir +'/ensemble_mask_overlays/%s'%(name), exist_ok=True)
            cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.png'%(name,name),image)
            cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.mask.png'%(name,name),color_overlay)
            cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.contour.png'%(name,name),contour_overlay)
            


        cv2.waitKey(0)
        # pass
def run_npy_to_sumbit_csv():

    image_dir   = '/data/steeve/DSB/data/image/stage1_test/images'

    submit_dir  = \
        out_dir + '/submit'


    npy_dir = submit_dir  + '/npys'
    csv_file = submit_dir + '/mask-rcnn-resnext-50-color_external130-iden-h-v-180-90.csv'

    ## start -----------------------------
    all_num=0
    cvs_ImageId = [];
    cvs_EncodedPixels = [];

    npy_files = glob.glob(npy_dir + '/*.npy')
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy','')

        multi_mask = np.load(npy_file)

        #<todo> ---------------------------------
        #post process here
        multi_mask = filter_small(multi_mask, 8)
        #<todo> ---------------------------------

        num = int( multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask==m+1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)
        all_num += num

        #<debug> ------------------------------------
#         print(all_num, num)  ##GT is 4152?
#         image_file = image_dir +'/%s.png'%name
#         image = cv2.imread(image_file)
#         color_overlay   = multi_mask_to_color_overlay(multi_mask)
#         color1_overlay  = multi_mask_to_contour_overlay(multi_mask, color_overlay)
#         contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])
#         all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
# #         image_show('all',all)
#         cv2.waitKey(1)


    #exit(0)
    # submission csv  ----------------------------

    # kaggle submission requires all test image to be listed!
    for t in ALL_TEST_IMAGE_ID:
        cvs_ImageId.append(t)
        cvs_EncodedPixels.append('') #null


    df = pd.DataFrame({ 'ImageId' : cvs_ImageId , 'EncodedPixels' : cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_ensemble()
    run_npy_to_sumbit_csv()
    print('\nsucess!')
