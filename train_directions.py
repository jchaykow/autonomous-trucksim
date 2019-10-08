from imports import *
from datasets import *
from models import *

class train_directions():
    def __init__(self, PATH = Path('data/')):
        self.PATH, self.IMG_PATH, self.CSV_PATH = PATH, PATH/'train3', PATH/'labels_directions.csv'
        arch = models.resnet34(pretrained=True)
        self.tensor_dataset = DirectionsDataset(
            csv_file=self.CSV_PATH, 
            root_dir=self.IMG_PATH,
            transform_img=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                transform_lab=transforms.Compose([
                    transforms.ToTensor()]))
        
        layer_list = list(arch.children())[-2:]
        arch = nn.Sequential(*list(arch.children())[:-2])
        arch.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        arch.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=layer_list[1].in_features, out_features=3, bias=True),
            normalize()
        )
        self.model = arch.to(device)
        self.dataset_size = len(tensor_dataset)
        self.dataloader = DataLoader(tensor_dataset, batch_size=16, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(arch.parameters(), lr=1e-2, momentum=0.9)
        # Decay LR by a factor of *gamma* every *step_size* epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        #self.directions = ['left', 'straight', 'right']

    def train_model(self, num_epochs=5):
        since = time.time()
        FT_losses = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        iters = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            self.exp_lr_scheduler.step()
            self.model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(self.dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                self.model.eval()   # Set model to evaluate mode
                with torch.no_grad():
                    outputs = self.model(inputs)
                    #set_trace()
                    _, preds = torch.max(outputs, 1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()

                FT_losses.append(loss.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #set_trace()
                iters += 1
                
                if iters % 2 == 0:
                    print('Prev Loss: {:.4f} Prev Acc: {:.4f}'.format(
                        loss.item(), torch.sum(preds == labels.data) / inputs.size(0)))

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, FT_losses
    
    def save(self, name:str):
        "Save model with `name` to `self.model_dir`."
        path = self.PATH + self.model_dir + f'{name}.pth'
        state = get_model(self.model).state_dict()
        torch.save(state, path)

