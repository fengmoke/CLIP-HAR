import torch
from clip_adapter import *
from TS2ACT import get_argparser
from rich.progress import Progress
import open_clip



torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def train_adapter(cfg, adapter_num):
    Train_epoch = 100
    backbone_clip = 'RN101'
    text_labels = {'UCI':['walking','walking upstairs','walking downstairs','sitting','standing','laying'],
                   'PAMAP2':['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping'],
                   'MotionSense': ['descending stairs', 'ascending stairs', 'walking', 'jogging', 'standing', 'sitting'],
                   'WISDM':['walking','jogging','ascending stairs','descending stairs','sitting','standing'],
                   'HHAR': ['biking', 'sitting', 'standing', 'walking', 'stair up', 'stair down']}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if adapter_num < 8 :
        model, preprocess = build_custom_clip(cfg.dataset, text_labels[cfg.dataset], backbone_clip, adapter_num)
    else:
        model, preprocess, visual_prompt = build_custom_clip(cfg.dataset, text_labels[cfg.dataset], backbone_clip, adapter_num)
        visual_prompt.to(device)
    # model, _, preprocess = open_clip.create_model_and_transforms(backbone_clip, pretrained='metaclip_400m', force_quick_gelu=True) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    if adapter_num > 7:
        optimizer = optim.Adam(visual_prompt.parameters(), lr=1e-4, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0) # 0: 1e-1, 0
    train_data = []
    val_data = []
    cur_state = random.getstate()
    print(f'当前的随机种子是{cur_state[1][0]}')
    for n, text_label in enumerate(text_labels[cfg.dataset]):
        selected_index = random.sample(list(range(50)), k=40)
        assert len(selected_index) == len(set(selected_index))
        data = []
        for i in range(50):
            data.append((preprocess(Image.open(f'TS2ACT-main/dataset/data/{cfg.dataset}/'+text_label+'/' + 
                str(i+1)+".jpg")), n))
            # if i in selected_index:
            #     train_data.append(data[i])
            # else:
            #     val_data.append(data[i])
            train_data.append(data[i])
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=8, shuffle=True,
                pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=8,
                pin_memory=False)
    best_loss = float('inf')
    Train_propress = Progress()
    Train_task = Train_propress.add_task("[green]Training", total=Train_epoch)
    for epoch in range(Train_epoch):
        Train_propress.start()
        train_acc = []
        if adapter_num > 7:
            model.eval()
            visual_prompt.train()
        else:
            model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        Loss = 0
        for i, (img, number_label) in enumerate(train_bar):
            img, number_label = img.to(device), number_label.to(device)
            optimizer.zero_grad()
            if adapter_num > 7:
                img = visual_prompt(img)
                output, _, _ = model(img)
            else:
                output, _, _ = model(img)
            train_acc.append((torch.argmax(output, dim=1) == number_label).float().mean())
            loss = criterion(output, number_label)
            Loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.desc = f'训练epoch {epoch+1}/{Train_epoch},Loss:{Loss / (i + 1):.3f}, train acc:{sum(train_acc) * 100 / len(train_acc):.3f}%'
        # model.eval()
        # val_acc = []
        # best_acc = 0
        # with torch.no_grad():
        #     for _ , (test_img, test_N_label) in enumerate(val_loader):
        #         test_img, test_N_label = test_img.to(device), test_N_label.to(device)
        #         test_output, _, _ = model(test_img)
        #         pre = torch.argmax(test_output, dim=1)
        #         val_acc.append((pre == test_N_label).float().mean())
        # print(f'Test Accuracy:{sum(val_acc) / len(val_acc):.3f}')
        backbone_clip = backbone_clip.replace('/', '_')
        if (Loss / (i + 1)) < best_loss:
            if adapter_num < 8:
                torch.save(model.state_dict(), f'TS2ACT-main/adapter/{cfg.dataset}_best_clip_adapter{adapter_num}_{backbone_clip}_1.pt')
            else:
                torch.save(visual_prompt.state_dict(), f'TS2ACT-main/adapter/{cfg.dataset}_best_clip_adapter{adapter_num}_{backbone_clip}.pt')
            best_acc = Loss / (i + 1)
        Train_propress.update(Train_task, advance=1)
        Train_propress.refresh()
    Train_propress.stop()

if __name__ == '__main__':
    opt = get_argparser().parse_args()
    train_adapter(opt, 0)