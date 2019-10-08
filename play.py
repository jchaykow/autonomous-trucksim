from imports import *
from key_presses import *

class play():
    def __init__(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        # bbox around truck
        self.warning = np.array([64, 64, 96, 64])
        # start with resnet34
        f_model = models.resnet34
        arch = models.resnet34(pretrained=True)
        # SSD arch
        k = 9 # len(anchor_scales)
        head_reg4 = SSD_MultiHead(k, -4.)
        self.ssd = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)
        # Directions arch
        layer_list = list(arch.children())[-2:]
        arch = nn.Sequential(*list(arch.children())[:-2])
        arch.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        arch.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=layer_list[1].in_features, out_features=3, bias=True),
            normalize()
        )
        self.directions = arch.to(device)
        # load SSD
        checkpointssd = torch.load('ssd.pth.tar')
        self.ssd.load_state_dict(checkpoint['state_dict'])
        # load Directions
        checkpoint = torch.load('directions.pth.tar')
        self.directions.load_state_dict(checkpoint['state_dict'])
    
    def main(self):
        '''
        This controls the video game autonomously.
        '''
        last_time = time.time()
        
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
        
        counter = 0
        with mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 79, "left": 265, "width": 905, "height": 586}

            while "Screen capturing":
                last_time = time.time()
                counter += 1
                # Get raw pixels from the screen, save it to a Numpy array
                screen = np.array(sct.grab(monitor))
                print('loop took {} seconds'.format(time.time()-last_time))
                last_time = time.time()
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                screen = cv2.resize(screen, (224,224)).astype(np.float32)/255
                '''
                ### for b/w images w/ lane detection ###

                stacked_img = np.stack((screen,)*3, axis=-1) 
                screen = cv2.cvtColor(stacked_img, cv2.COLOR_BGR2RGB)
                '''
                self.directions.eval()
                img = renorm(V(np.transpose(screen[None], (0,3,1,2))))
                log_pred = to_np(self.directions(img))
                #log_pred = learn.predict_array(img[None])
                moves = np.around(np.exp(log_pred))
                print('Here are the moves:', moves)

                # run object detection model
                self.ssd.eval()
                b_clas_truck,b_bb_truck = self.ssd(img)
                a_ic_truck = actn_to_bb(b_bb_truck[0], anchors)
                clas_pr_truck, clas_ids_truck = b_clas_truck[0].max(1)
                clas_pr_truck = clas_pr_truck.sigmoid()

                bbox = to_np((a_ic_truck*224).long())
                bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
                print('Here is the bb:', bb[0])

                clas = clas_ids_truck
                prs = clas_pr_truck
                thresh = 0.15

                # loop through each bounding box in the frame and calculate if it overlaps with the area around the truck
                if prs is None:  prs  = [None]*len(bb)
                if clas is None: clas = [None]*len(bb)
                move_warning = np.array([0,0,0])
                for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
                    c = float(to_np(c))
                    pr = float(to_np(pr))
                    if((b[2]>0) and (pr is None or pr > thresh)):
                        move_warning = move_warning + overlapping2D(b)
                        cv2.rectangle(screen, tuple(b[:2]), tuple(b[:2]+b[-2:]), (0,0,255), 1)
                        txt = id2cat[int(c)]
                        cv2.putText(screen,txt,tuple(b[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,155), 2, cv2.LINE_AA)

                # Display the picture
                cv2.imwrite(f'data/record/screen{counter}.png',cv2.cvtColor(screen, cv2.COLOR_RGB2BGR) * 255)
                print('Here is the move-warning:', np.argmax(move_warning), move_warning)
                print(directions[np.argmax(log_pred)])

                if (moves == [1,0,0]).all():
                    if np.sum(move_warning) != 0:
                        warning = self.convert_warnings(move_warning)
                        if warning == 'right':
                            right()
                        if warning == 'straight':
                            straight()
                    left()
                elif (moves == [0,1,0]).all():
                    if np.sum(move_warning) != 0:
                        warning = self.convert_warnings(move_warning)
                        if warning == 'left':
                            left()
                        if warning == 'right':
                            right()
                    straight()
                elif (moves == [0,0,1]).all():
                    if np.sum(move_warning) != 0:
                        warning = self.convert_warnings(move_warning)
                        if warning == 'left':
                            left()
                        if warning == 'straight':
                            straight()
                    right()
                else:
                    if np.sum(move_warning) != 0:
                        slow_ya_roll()
                        warning = self.convert_warnings(move_warning)
                        if warning == 'left':
                            left()
                        if warning == 'straight':
                            straight()
                        if warning == 'right':
                            right()

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    
    def overlapping2D(self, box_a): 
        xmin1, xmax1 = (box_a[0], box_a[0] + box_a[2])
        xmin2, xmax2 = (self.warning[0], self.warning[0] + self.warning[2])
        
        ymin1, ymax1 = (box_a[1], box_a[1] + box_a[3])
        ymin2, ymax2 = (self.warning[1], self.warning[1] + self.warning[3])
        
        check1Dx = xmax1 >= xmin2 and xmax2 >= xmin1
        
        check1Dy = ymax1 >= ymin2 and ymax2 >= ymin1
        
        if check1Dx and check1Dy and ((xmin1 + xmax1) / 2) < 112:
            return np.array([0,0,1])
        if check1Dx and check1Dy and ((xmin1 + xmax1) / 2) > 112:
            return np.array([1,0,0])
        else:
            return np.array([0,0,0])

    def convert_warnings(self):
        directions = ['left', 'straight', 'right']
        return directions[np.argmax(self.warning)]

    def draw_bboxes(self, img, bboxes, color=(0, 0, 255), thickness=1):
        for bbox in bboxes:
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[:2]+bbox[-2:]), color, thickness)
