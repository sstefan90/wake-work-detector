import torch
import torch.nn as nn
import numpy as np
from cnn import AudioModel
from torch.utils.mobile_optimizer import optimize_for_mobile

MODEL_PATH = "logs/audiomodel.batch_size:16.lr:0.0005.lr_schedule:5.weight_decay:0.1.weight:4.0.epochs:30/checkpoint/checkpoint_25.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AudioModelPTQ(nn.Module):

    def __init__(self):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5,5), padding='valid', stride=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_1 = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU()

        self.conv_block_1 = nn.Sequential( self.conv_1, self.pool_1, self.bn_1, self.relu)
        
        self.conv_2 = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=(3,3), padding='valid', stride=1)
        self.pool_2= nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_2 = nn.BatchNorm2d(num_features=2)

        self.conv_block_2 = nn.Sequential(self.conv_2, self.pool_2, self.bn_2, self.relu)

        self.conv_3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3,3), padding='valid')
        self.pool_3= nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_3 = nn.BatchNorm2d(num_features=1)

        self.conv_block_3 = nn.Sequential(self.conv_3, self.pool_3, self.relu)

        self.fc_1 = nn.Linear(in_features=864, out_features=64)
        self.fc_2 = nn.Linear(in_features=64, out_features=2)

        self.linear_layer = nn.Sequential(self.fc_1, self.relu, self.fc_2)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self):
        x = self.quant(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_3(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear_layer(x)
        x = self.dequant(x)
        return x


def model_checkpoint(model, log_dir):
    checkpoint = {
        'model': model.state_dict(),
    }
    torch.save(checkpoint, f'{log_dir}/quantized_model.pth')

def model_size():
    model= AudioModel()
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model'])
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

'''
def quantization_aware_training():
    shape_file = "data_processed/test_shape.txt"
    with open(shape_file, 'r') as f:
        for line in f:
            shape = [int(x) for x in line.strip().split(",")]
            shape = tuple(shape)

    calibration_dataset_path = "data_processed/X_test.dat"
    calibration_dataset = torch.from_numpy(np.memmap(f'{calibration_dataset_path}', dtype='float32', mode='r', shape=(shape[0], 1)))

    model_q = AudioModelPTQ()
    checkpoint = torch.load(MODEL_PATH)
    model_q.load_state_dict(checkpoint['model'])
    model_q.eval()
    model_q = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    model_q_fused = torch.ao.quantization.fuse_modules(model_q, [['bn', 'relu']])
    model_q_prepared = torch.ao.quantization.prepare(model_q_fused.train())
    model_q_prepared(calibration_dataset)
    model_int8 = torch.ao.quantization.convert(model_q_prepared)
    return model_int8

def post_quantization():
    pass

def training_loop():
    progress_bar = tqdm.tqdm(range(args.epochs))
    training_dataloader, val_dataloader = create_dataloader(args.batch_size)
    model = AudioModel()
    model.to(DEVICE)
    model.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0, args.weight]))

    for epoch in progress_bar:

        step = 0
        train_loss = []
        for X_train, y_train in training_dataloader:
            X_train.to(DEVICE)
            y_train.to(DEVICE)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
            y_train = nn.functional.one_hot(y_train, num_classes=2).view(y_train.shape[0], 2).float()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iteration = epoch*(len(training_dataloader)) + step
            train_loss.append(loss.detach().cpu().item())
            step +=1
        scheduler.step()
        
    
        #at the end of each epoch, run validation set
        with torch.inference_mode():
            val_loss = []
            for X_val, y_val in val_dataloader:
                X_val.to(DEVICE)
                y_val.to(DEVICE)

                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
                y_val= nn.functional.one_hot(y_val, num_classes=2).view(y_val.shape[0], 2).float()

                logits = model(X_val)
                loss = criterion(logits, y_val)
                val_loss.append(loss.detach().cpu().item())

        writer.add_scalars(f'loss', {
            'train_loss': sum(train_loss)/len(train_loss),
            'val_loss': sum(val_loss)/ len(val_loss)
        }, iteration)

        print(f'epoch {epoch}, train_loss: {sum(train_loss) / len(train_loss)}, val_loss: {sum(val_loss)/ len(val_loss)}')

        #checkpoint the model
        model_checkpoint(model,epoch, optimizer, scheduler, log_dir)
'''

def create_mobile_model():
    #pass
    #process model to make it runnable on c++
    model= AudioModel()
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model'])

    dummy_input = torch.rand([1, 1, 128,147])
    torchscript_model = torch.jit.trace(model, dummy_input)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "quantized_model/audiomodel_mobile.pt")



if __name__ == "__main__":
    #model = create_quantized_model()
    #model_checkpoint(model, "quantized_model")
    model_size()
    create_mobile_model()






