########################
# Section I: read / load
########################
def read_image(path):
    """
    Read image from path, convert to RGB channel.
    """
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def get_im_bb(path, df):
    """
    Get single image and bbox based on path and df.
    """
    im = read_image(path)
    f_name = path.parts[-1]
    bb = df.loc[df.fname == f_name, ["x_min", "y_min", "x_max", "y_max"]].values.squeeze()
    return im, bb


#########################
# Section II: plot / show
#########################
def create_corner_rect(bb, color='red'):
    """
    Create a red bbox for plotting.
    """
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    """
    Plot image with bbox.
    """
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    
    
##########################
# Section III: bbox2pixels
##########################
def make_bb_px(y, x):
    """
    Makes an image of size x retangular bounding box.
    Represent bbox using image.
    """
    r,c,*_ = x.shape
    Y = np.zeros((r, c))
    y = y.astype(np.int)
    Y[y[1]:y[3], y[0]:y[2]] = 1.
    return Y

def to_bb(Y):
    """
    Convert mask Y to a bounding box, assumes 0 as background nonzero object.
    Get bbox from the image representation.
    """
    rows, cols = np.nonzero(Y)
    if len(cols)==0: return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


####################
# Section IV: resize
####################
def resize_tr(im, bb, sz):
    """
    Resize image and bbox at the same time.
    """
    Y = make_bb_px(bb, im)
    im2 = cv2.resize(im, (sz, sz))
    Y2 = cv2.resize(Y, (sz, sz))
    return im2, to_bb(Y2)


def resize_all_images(df, train_path, valid_path, org_path, sz=224):
    """
    Resize all images and store in folders.
    """
    np.random.seed(3)
    tmp = df.copy()
    files = list(org_path.iterdir())
    
    for f in files:
        f_name = f.parts[-1]
        if np.random.uniform() < 0.8:
            new_path = train_path/f_name
        else:
            new_path = valid_path/f_name
        im = read_image(f)
        bb = tmp.loc[tmp.fname == f_name, ["x_min", "y_min", "x_max", "y_max"]].values.squeeze()  # case where no bb
        im, bb = resize_tr(im, bb, sz)
        
        cv2.imwrite(str(new_path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        tmp.loc[tmp.fname == f_name, ["x_min", "y_min", "x_max", "y_max"]] = bb
        tmp.to_csv(car_annos/"annos_224.csv", index=False)
        

#########################
# Section V: augmentation
#########################
def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """
    Rotates an image (bbox image) by deg degrees.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)  # center, angle, scale
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)


def crop(im, r, c, target_r, target_c): 
    """
    Crop images.
    """
    return im[r:r+target_r, c:c+target_c]


def random_cropXY(x, Y, r_pix=8):
    """
    Returns a random crop. Dimensions are kept the same.
    """
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY


def center_crop(x, r_pix=8):
    """
    Do a center crop.
    """
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


def transformsXY(path, df, transforms): 
    """
    Wrapper of rotation, cropping to do transformations.
    """
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    f_name = path.parts[-1]
    bb = df.loc[df.fname == f_name, ["x_min", "y_min", "x_max", "y_max"]].values.squeeze()
    Y = make_bb_px(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20  # -10 -- 10
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
#     else:
#         x, Y = center_crop(x), center_crop(Y)
    return x, to_bb(Y)


#####################
# Section VI: dataset
#####################
def normalize(im):
    """
    Normalizes images with Imagenet stats.
    """
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


class CarDataset(Dataset):
    """
    Example: car images with bboxes.
    """
    def __init__(self, path, df, transforms=False):
        self.files = list(path.iterdir())
        self.df = df
        self.transforms = transforms
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        f_name = path.parts[-1]
        y_class = self.df.loc[self.df.fname == f_name, "class"].values[0] - 1  # 0 indexed

        bb_exist = 0
        if f_name in list(self.df.fname):
            bb_exist = 1
        
        x, y_bb = transformsXY(path, self.df, self.transforms)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb, bb_exist
    

class CarNet(nn.Module):
    """
    Example: predict 196 classes along with bboxes.
    """
    def __init__(self):
        super(CarNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier1 = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 196))
        self.classifier2 = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
    
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2

    
#######################
# Section VII: training
#######################
def cosine_segment(start_lr, end_lr, iterations):
    """
    Create segments of learning rates.
    """
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 *c_i


def get_cosine_triangular_lr(max_lr, iterations):
    """
    Create a cyclical learning rate curve.
    """
    min_start, min_end = max_lr/25, max_lr/(25*1e4)
    iter1 = int(0.3*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, max_lr, iter1), cosine_segment(max_lr, min_end, iter2)]
    return np.concatenate(segs)


def create_optimizer(model, lr0):
    """
    Set different learning rate at different layers.
    """
    params = [{'params': model.features1.parameters(), 'lr': lr0/9},
              {'params': model.features2.parameters(), 'lr': lr0/3},
              {'params': model.classifier1.parameters(), 'lr': lr0},
              {'params': model.classifier2.parameters(), 'lr': lr0}]
    return optim.Adam(params, weight_decay=1e-5)


def update_optimizer(optimizer, group_lrs):
    """
    Update the model optimizer.
    """
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = group_lrs[i]

        
def save_model(m, p): torch.save(m.state_dict(), p)
    
def load_model(m, p): m.load_state_dict(torch.load(p))

        
def LR_range_finder(model, train_dl, lr_low=1e-5, lr_high=0.05, epochs=2):
    """
    Find the optimal learning rate.
    """
    losses = []
    p = PATH/"mode_tmp.pth"
    save_model(model, str(p))
    iterations = epochs * len(train_dl)
    delta = (lr_high - lr_low)/iterations
    lrs = [lr_low + i*delta for i in range(iterations)]
    optimizer = create_optimizer(model, lrs[0])
    model.train()
    ind = 0
    for i in range(epochs):
        for x,y in train_dl:
            lr = lrs[ind]
            update_optimizer(optimizer, [lr/9, lr/3, lr])
            x = x.cuda().float()
            y = y.cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            ind +=1
           
    load_model(model, str(p))
    return lrs, losses


def val_metrics(model, valid_dl, C=1000):
    """
    Evaluate the validation set.
    """
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y_class, y_bb, z in valid_dl:
        batch = y_class.shape[0]
        x = x.float().cuda()
        y_class = y_class.long().cuda()
        y_bb = y_bb.float().cuda()
        z = z.float().cuda()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = z*F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)  # pred are indices
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total

        
def train_triangular_policy(model, train_dl, valid_dl, max_lr=0.01, epochs=10, C=1000):
    """
    Train the model.
    """
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_cosine_triangular_lr(max_lr, iterations)
    optimizer = create_optimizer(model, lrs[0])
    prev_val_acc = 0.0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb, z in train_dl:
            lr = lrs[idx]
            update_optimizer(optimizer, [lr/9, lr/3, lr, lr])
            batch = y_class.shape[0]
            x = x.float().cuda()
            y_class = y_class.long().cuda()
            y_bb = y_bb.float().cuda()
            z = z.float().cuda()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = z*F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        if val_acc > prev_val_acc: 
            prev_val_acc = val_acc
            if val_acc > 0.8:
                path = "models/model_resnet34_loss_{0:.0f}.pth".format(100*val_acc)
                save_model(model, path)
                print(path)
    return sum_loss/total