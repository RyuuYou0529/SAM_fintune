import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def sample_points(img, low_num=100, high_num=200):
    img = np.asarray(img)

    front_point = []
    front_label = []
    back_point = []
    back_label = []

    h, w = img.shape
    front_num = np.random.randint(low_num,high_num+1)
    while front_num!=0:
        h_index = np.random.randint(0, h)
        w_index = np.random.randint(0, w)
        if img[h_index, w_index]==255:
            front_point.append([w_index, h_index])
            front_label.append(1)
            front_num-=1

    back_num = np.random.randint(low_num,high_num+1)
    while back_num!=0:
        h_index = np.random.randint(0, h)
        w_index = np.random.randint(0, w)
        if img[h_index, w_index]==0:
            back_point.append([w_index, h_index])
            back_label.append(0)
            back_num-=1

    point_list = np.asarray(front_point+back_point)
    label_list = np.asarray(front_label+back_label)

    state = np.random.get_state()
    np.random.shuffle(point_list)
    np.random.set_state(state)
    np.random.shuffle(label_list)

    point_list = torch.from_numpy(point_list).to(torch.float)
    label_list = torch.from_numpy(label_list).to(torch.float)

    return point_list, label_list

class DRIVE_Dataset(Dataset):
    def __init__(self, input_folder, label_folder):
        self.input_list = [os.path.join(input_folder, name) for name in sorted(os.listdir(input_folder))]
        self.label_list = None if label_folder is None else [os.path.join(label_folder, name) for name in sorted(os.listdir(label_folder))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        input = Image.open(self.input_list[index])
        label = Image.open(self.label_list[index])
        points = sample_points(label)
        return self.transform(input), self.transform(label), points
    
    def __len__(self):
        return len(self.input_list)

# ============================
# Loss
# ============================
def dice_coefficient(y_pred, y_true):
    eps = 1.0
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def dice_loss(y_pred, y_true):
    return 1 - dice_coefficient(y_pred, y_true)

if __name__ == '__main__':
    # ==========
    # config
    # ==========
    num_epoch = 100
    batch_size=4
    lr = 1e-4
    device = torch.device('cuda:0')
    checkpoint_path = '/share/home/liuy/project/SAM_finetune/checkpoints/finetune'
    model_save_path = os.path.join(checkpoint_path, 'best.pth')

    # ==========
    # data
    # ==========
    input_folder='/share/home/liuy/project/SAM_finetune/data/DRIVE/training/images'
    label_folder='/share/home/liuy/project/SAM_finetune/data/DRIVE/training/1st_manual'

    ds = DRIVE_Dataset(input_folder=input_folder, label_folder=label_folder)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    # ==========
    # prepare model
    # ==========
    pretrained_path = '/share/home/liuy/project/SAM_finetune/checkpoints/sam_vit_b_01ec64.pth'
    model_type = "vit_b"

    sam_model = sam_model_registry[model_type]()
    sam_model.to(device=device)

    sam_state = sam_model.state_dict()
    pretrained_state = torch.load(pretrained_path)
    state_dict = {k: v for k, v in pretrained_state.items() if k in sam_state.keys()}

    sam_state.update(state_dict)
    sam_model.load_state_dict(sam_state)
    # ==========
    # optimizer & loss
    # ==========
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # ==========
    # loss
    # ==========
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    best_loss = 1000

    # ==========
    # tensorboard
    # ==========
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'tensorboard/'))
    
    train_total_iters = 0
    for i in range(num_epoch):

        loop = tqdm(enumerate(dl), ncols=100, total=len(dl))
        loop.set_description(f'Epoch [{i+1}/{num_epoch}]')

        for index, (inputs, _, (points, labels)) in loop:
            train_total_iters += batch_size

            inputs = inputs.to(device)
            # masks = masks.to(device)
            points = points.to(device)
            labels = labels.to(device)

            original_size = inputs.shape[-2:]
            inputs = resize_transform.apply_image_torch(inputs)
            points = resize_transform.apply_coords_torch(points, original_size)

            inputs = torch.stack([sam_model.preprocess(x) for x in inputs], dim=0)

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(inputs)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )
            
            low_res_masks, iou_predictions, point_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            labels = labels.unsqueeze(1)
            loss = loss_fn(point_predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/BCE', loss, global_step=train_total_iters)

            loop.set_postfix_str(f'loss={loss.item():.6f}')
            
            if loss.item() < best_loss:
                print(f'checkpoint at [epoch {i+1}/{num_epoch}][batch {index}/{len(dl)}]')
                torch.save(sam_model.state_dict(), model_save_path)