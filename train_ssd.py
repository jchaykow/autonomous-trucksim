from imports import *
from datasets import *
from models import *

class train_ssd():
    def __init__(self):
        PATH_pascal = Path('data/pascal')
        trn_j = json.load((PATH_pascal / 'pascal_train2007.json').open())
        IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']
        FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'
        cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
        trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
        trn_ids = [o[ID] for o in trn_j[IMAGES]]
        # JPEGS_pascal = 'VOCdevkit2/VOC2007/JPEGImages'
        # IMG_PATH_pascal = PATH_pascal/JPEGS_pascal

        trn_anno = get_trn_anno()
        self.id2cat = list(cats.values())
        anc_grids = [4,2,1]
        anc_zooms = [0.7, 1., 1.3]
        anc_ratios = [(1.,1.), (1.,0.5), (0.5,1.)]
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)

        anc_sizes = np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = V(np.concatenate([np.array([1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids]), requires_grad=False).unsqueeze(1)
        anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        n_clas = len(self.id2cat)+1
        n_act = k*(4+n_clas)
        MC_CSV = PATH_pascal/'tmp/mc.csv'
        CLAS_CSV = PATH_pascal/'tmp/clas.csv'
        MBB_CSV = PATH_pascal/'tmp/mbb.csv'
        mc = [[cats[p[1]] for p in trn_anno[o]] for o in trn_ids]
        cat2id = {v:k for k,v in enumerate(self.id2cat)}
        mcs = np.array([np.array([cat2id[p] for p in o]) for o in mc])

        mbb = [np.concatenate([p[0] for p in trn_anno[o]]) for o in trn_ids]
        mbbs = [' '.join(str(p) for p in o) for o in mbb]

        #mbb = pd.read_csv('data/pascal/tmp/mbb.csv')
        #mc = pd.read_csv('data/pascal/tmp/mc.csv')

        self.bb_dataset = BboxDataset(
            csv_file='data/pascal/tmp/mbb.csv',
            root_dir='data/pascal/VOCdevkit2/VOC2007/JPEGImages/',
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        
        self.unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        num_colr = 14
        cmap = get_cmap(num_colr)
        colr_list = [cmap(float(x)) for x in range(num_colr)]

        self.trn_ds2 = ConcatLblDataset(self.bb_dataset, mcs)
        self.bb_dataloader2 = DataLoader(self.trn_ds2, batch_size=16, shuffle=True)

        self.trn_ds2k = len(anchor_scales)
        head_reg4 = SSD_MultiHead(k, -4.)
        f_model = models.resnet34
        modelss = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)
        self.model = modelss.model

        beta1 = 0.5
        self.optimizer = optim.Adam(modelss.model.parameters(), lr=1e-3, betas=(beta1, 0.99))
        self.exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    def train_ssd_model(self, num_epochs=5):
        since = time.time()
        SSD_losses = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 0.0
        iters = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            self.exp_lr_scheduler.step()
            self.model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(self.bb_dataloader2):
                inputs = inputs.to(device)
                bbs = labels[0].to(device)
                labels = labels[1].to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                self.model.eval()
                outputs = self.model(inputs)
                loss = self.ssd_loss(outputs, [bbs,labels], False)

                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()

                SSD_losses.append(loss.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                #set_trace()
                iters += 1
                
                if iters % 2 == 0:
                    print('Prev Loss: {:.4f}'.format(loss.item()))
            
            epoch_loss = running_loss / dataset_size
            print('Loss: {:.4f}'.format(epoch_loss))

            # deep copy the model
            if epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, SSD_losses
    

    def test_ssd(self):
        x,y = next(iter(self.bb_dataloader2))
        y = V(y)
        b_clas_truck,b_bb_truck = modelss.model(V(x))
        ima = np.transpose(unorm(x), (0,2,3,1))[0]
        ax = plt.gca()
        bbox,clas = get_y(y[0][0], y[1][0])
        a_ic_truck = actn_to_bb(b_bb_truck[0], anchors)
        clas_pr_truck, clas_ids_truck = b_clas_truck[0].max(1)
        clas_pr_truck = clas_pr_truck.sigmoid()
        torch_gt(ax, ima, a_ic_truck, clas_ids_truck, clas_pr_truck, clas_pr_truck.max().data.item()*0.75)
        plt.tight_layout()
    

    def save(self, name:str):
        "Save model with `name` to `self.model_dir`."
        path = self.PATH + self.model_dir + f'{name}.pth'
        state = get_model(self.model).state_dict()
        torch.save(state, path)
    

    def loss_f(self):
        return BCE_Loss(len(self.id2cat))
    

    def ssd_1_loss(self,b_c,b_bb,bbox,clas,print_it=False):
        bbox,clas = get_y(bbox,clas)
        a_ic = actn_to_bb(b_bb, anchors)
        overlaps = jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(id2cat)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss
    

    def ssd_loss(self,pred,targ,print_it=False):
        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas,print_it)
            lls += loc_loss
            lcs += clas_loss
        if print_it: 
            print(f'loc: {lls.data.item()}, clas: {lcs.data.item()}')
        return lls+lcs
