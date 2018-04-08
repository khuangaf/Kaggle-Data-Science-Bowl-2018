from common import *
from dataset.transform import *
from net.layer.mask_nms import *

##--------------------------------------------------------------
AUG_FACTOR = 16
AUG_SCALE = 0.5

## argument ##########################################


def make_mask_more(net):

    detection  = net.detections.data.cpu().numpy()
    mask_logit = net.mask_logits.cpu().data.numpy()
    mask_prob  = np_sigmoid(mask_logit)
    mask       = net.masks[0]

    height,width = mask.shape[:2]
    mask_score = np.zeros((height,width),np.float32)
    num_detection = len(detection)
    for n in range(num_detection):
        _,x0,y0,x1,y1,score,label,k = detection[n]
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        label = int(label)
        k = int(k)
        h, w  = y1-y0+1, x1-x0+1


        crop  = mask_prob[k, label]
        crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
        mask_score[y0:y1+1,x0:x1+1] += crop

    return mask_score


def do_test_augment_identity(image):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )
    return image


def undo_test_augment_identity(net, image):

    height,width = image.shape[:2]
    # h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    # w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    # dx = w-width
    # dy = h-height

    rcnn_proposal = net.rcnn_proposals.cpu().numpy()
    detection = net.detections.data.cpu().numpy()
    mask =  net.masks[0]


    # rcnn_proposal[:,1]=np.clip(rcnn_proposal[:,1]-AUG_BORDER,0,width -1)
    # rcnn_proposal[:,2]=np.clip(rcnn_proposal[:,2]-AUG_BORDER,0,height-1)
    # rcnn_proposal[:,3]=np.clip(rcnn_proposal[:,3]-AUG_BORDER,0,width -1)
    # rcnn_proposal[:,4]=np.clip(rcnn_proposal[:,4]-AUG_BORDER,0,height-1)
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = max(min(x0,width -1),0)
        x1 = max(min(x1,width -1),0)
        y0 = max(min(y0,height-1),0)
        y1 = max(min(y1,height-1),0)
        w = x1-x0 + 1
        h = y1-y0 + 1

        if w>2 and h>2:
            rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])

    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    # detection[:,1]=np.clip(detection[:,1]-AUG_BORDER,0,width -1)
    # detection[:,2]=np.clip(detection[:,2]-AUG_BORDER,0,height-1)
    # detection[:,3]=np.clip(detection[:,3]-AUG_BORDER,0,width -1)
    # detection[:,4]=np.clip(detection[:,4]-AUG_BORDER,0,height-1)
    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = max(min(x0,width -1),0)
        x1 = max(min(x1,width -1),0)
        y0 = max(min(y0,height-1),0)
        y1 = max(min(y1,height-1),0)
        w = x1-x0 + 1
        h = y1-y0 + 1
        if w>2 and h>2:
            detection.append([i,x0,y0,x1,y1, score, label, aux])
        else:
            mask[mask==t+1]=0

    detection= np.array(detection, np.float32)


    mask  = mask[0:height,0:width]
    mask_score = make_mask_more(net)[0:height,0:width]
    #predict = relabel_mask(predict)

    return rcnn_proposal, detection, mask, mask_score


'''
def randomTransposeAndFlip(img, u=None):
    if u is None:
        u = random.randint(0,7)  #choose one of the 8 cases

    if u==1: #rotate90
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)
    if u==2: #rotate180
        img = cv2.flip(img,-1)
    if u==3: #rotate270
        img = img.transpose(1,0,2)
        img = cv2.flip(img,0)

    if u==4: #flip left-right
        img = cv2.flip(img,1)
    if u==5: #flip up-down
        img = cv2.flip(img,0)
        
    if u==6:
        img = cv2.flip(img,1)
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)

    if u==7:
        img = cv2.flip(img,0)
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)

    return img
'''

## argument ##########################################

def do_test_augment_horizontal_flip(image):
    image = cv2.flip(image,1)
    image = do_test_augment_identity(image)
    return image



def undo_test_augment_horizontal_flip(net, image):
    rcnn_proposal, detection, mask, mask_score = undo_test_augment_identity(net, image)


    height,width = mask.shape[:2]
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = width -1 - x0
        x1 = width -1 - x1
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = width -1 - x0
        x1 = width -1 - x1
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection= np.array(detection, np.float32)

    mask = np.fliplr(mask)
    mask_score = np.fliplr(mask_score)
    return rcnn_proposal, detection, mask, mask_score



## argument ##########################################

def do_test_augment_vertical_flip(image):
    image = cv2.flip(image,0)
    image = do_test_augment_identity(image)
    return image



def undo_test_augment_vertical_flip(net, image):
    rcnn_proposal, detection, mask, mask_score = undo_test_augment_identity(net, image)


    height,width = mask.shape[:2]
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        y0 = height -1 - y0
        y1 = height -1 - y1
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        y0 = height -1 - y0
        y1 = height -1 - y1
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection= np.array(detection, np.float32)

    mask = np.flipud(mask)
    mask_score = np.flipud(mask_score)
    return rcnn_proposal, detection, mask, mask_score

## argument ##########################################

def do_test_augment_rotate090(image):
    image = image.transpose(1,0,2)  #cv2.transpose(img)
    image = cv2.flip(image,1)
    image = do_test_augment_identity(image)
    return image



def undo_test_augment_rotate090(net, image):
    image = image.transpose(1,0,2)
    rcnn_proposal, detection, mask, mask_score = undo_test_augment_identity(net, image)


    height,width = mask.shape[:2]
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0,y0 = y0,width-1-x0
        x1,y1 = y1,width-1-x1
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0,y0 = y0,width-1-x0
        x1,y1 = y1,width-1-x1
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection= np.array(detection, np.float32)

    mask = np.fliplr(mask)
    mask = mask.transpose(1,0)  #cv2.transpose(img)
    mask_score = np.fliplr(mask_score)
    mask_score = mask_score.transpose(1,0)  #cv2.transpose(img)
    return rcnn_proposal, detection, mask, mask_score


## argument ##########################################

def do_test_augment_rotate180(image):
    image = cv2.flip(image,-1)
    image = do_test_augment_identity(image)
    return image



def undo_test_augment_rotate180(net, image):
    rcnn_proposal, detection, mask, mask_score = undo_test_augment_identity(net, image)


    height,width = mask.shape[:2]
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = width  -1 - x0
        x1 = width  -1 - x1
        y0 = height -1 - y0
        y1 = height -1 - y1
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = width  -1 - x0
        x1 = width  -1 - x1
        y0 = height -1 - y0
        y1 = height -1 - y1
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection= np.array(detection, np.float32)

    mask = np.fliplr(mask)
    mask = np.flipud(mask)
    mask_score = np.fliplr(mask_score)
    mask_score = np.flipud(mask_score)
    return rcnn_proposal, detection, mask, mask_score


## argument ##########################################

def do_test_augment_rotate270(image):
    image = image.transpose(1,0,2) #cv2.transpose(img)
    image = cv2.flip(image,0)
    image = do_test_augment_identity(image)
    return image



def undo_test_augment_rotate270(net, image):
    image = image.transpose(1,0,2)
    rcnn_proposal, detection, mask, mask_score = undo_test_augment_identity(net, image)


    height,width = mask.shape[:2]
    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0,y0 = height-1-y0,x0
        x1,y1 = height-1-y1,x1
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0,y0 = height-1-y0,x0
        x1,y1 = height-1-y1,x1
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection= np.array(detection, np.float32)

    mask = np.flipud(mask)
    mask = mask.transpose(1,0)  #cv2.transpose(img)
    mask_score = np.flipud(mask_score)
    mask_score = mask_score.transpose(1,0)
    return rcnn_proposal, detection, mask, mask_score



## argument ##########################################

def do_test_augment_scale(image, scale_x=AUG_SCALE, scale_y=AUG_SCALE):
    image = scale_to_factor(image, scale_x, scale_y, factor=AUG_FACTOR)
    return image


def undo_test_augment_scale(net, image):


    def scale_mask(detection, mask_prob, width, height ):
        mask_threshold=0.5

        mask       = np.zeros((height,width),np.int32)
        mask_score = np.zeros((height,width),np.float32)
        num_detection = len(detection)
        for n in range(num_detection):
            _,x0,y0,x1,y1,score,label,k = detection[n]
            x0 = int(round(x0))
            y0 = int(round(y0))
            x1 = int(round(x1))
            y1 = int(round(y1))
            label = int(label)
            k = int(k)
            h, w  = y1-y0+1, x1-x0+1

            crop  = mask_prob[k, label]
            crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)

            mask[y0:y1+1,x0:x1+1][np.where(crop > mask_threshold)] = n+1
            mask_score[y0:y1+1,x0:x1+1] += crop

        ## post process -------------------- <todo> this is in complete code
        if 1:
            mask = remove_small_fragment(mask, min_area=5)
            #mask = fill_hole(mask)

        ##remove small fragments --------------------

        return mask, mask_score



    # ----
    rcnn_proposal = net.rcnn_proposals.cpu().numpy()
    detection = net.detections.data.cpu().numpy()
    mask =  net.masks[0]


    mask_logit = net.mask_logits.cpu().data.numpy()
    mask_prob  = np_sigmoid(mask_logit)


    height,width = image.shape[:2]
    H,W  = mask.shape[:2]
    scale_x = width/W
    scale_y = height/H

    ps = rcnn_proposal.copy()
    rcnn_proposal=[]
    for p in ps:
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = min(int(round(x0*scale_x)),width-1)
        y0 = min(int(round(y0*scale_y)),height-1)
        x1 = min(int(round(x1*scale_x)),width-1)
        y1 = min(int(round(y1*scale_y)),height-1)
        rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    rcnn_proposal= np.array(rcnn_proposal, np.float32)


    ps = detection.copy()
    detection=[]
    for t,p in enumerate(ps):
        i,x0,y0,x1,y1, score, label, aux = p
        x0 = min(int(round(x0*scale_x)),width-1)
        y0 = min(int(round(y0*scale_y)),height-1)
        x1 = min(int(round(x1*scale_x)),width-1)
        y1 = min(int(round(y1*scale_y)),height-1)
        detection.append([i,x0,y0,x1,y1, score, label, aux])
    detection = np.array(detection, np.float32)

    mask, mask_score = scale_mask(detection, mask_prob, width, height )

    return rcnn_proposal, detection, mask, mask_score






