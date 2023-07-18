import SimpleITK as sitk
import numpy as np
import scipy.ndimage as sp

def get_bounding_box(img,seg,x,y,z):
    # seg is a 3D array consisting of 0 or 1
    coords = np.argwhere(seg)
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    dx = (x-xmax+xmin)//2
    dy = (y-ymax+ymin)//2
    dz = (z-zmax+zmin)//2
    return img[xmin-dx:xmin-dx+x,ymin-dy:ymin-dy+y,zmin-dz:zmin-dz+z]

def get_bounding_edge(img,seg,x,y,z):
    # seg is a 3D array consisting of 0 or 1
    coords = np.argwhere(seg)
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    dx = (x-xmax+xmin+10)//4
    dy = (y-ymax+ymin+10)//4
    dz = (z-zmax+zmin+10)//4
    return img[xmin-dx:xmin-dx+x,ymin-dy:ymin-dy+y,zmin-dz:zmin-dz+z]


def crop_3D(img):
    # seg is a 3D array consisting of 0 or 1
    coords = np.argwhere(img)
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    return img[xmin:xmax,ymin:ymax,zmin:zmax]

def crop_3D_cutting(img):
    # seg is a 3D array consisting of 0 or 1
    coords = np.argwhere(img)
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    return img[xmin:xmax,ymin:ymax,zmin:zmax]

def crop_3D_2(img1,img2):
    # seg is a 3D array consisting of 0 or 1
    coords = np.argwhere(img1)
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    a = img1[xmin:xmax, ymin:ymax, zmin:zmax]
    b = img2[xmin:xmax,ymin:ymax,zmin:zmax]
    return a,b

def pad256(array):
    print("original array.shape",array.shape)
    if array.shape[1]<256:
        dy1 = round(256-array.shape[1])
        if dy1 % 2 == 0:
            print("before pad y ",array.shape)
            img = np.pad(array, ((0, 0),(round((dy1/2)), round(dy1/2)), (0, 0)), 'constant')
            print("after pad y ",img.shape)
        else:
            print("before pad y ",array.shape)
            img = np.pad(array, ((0, 0), (round((dy1/2)), round((dy1/2)+0.5)) , (0, 0)), 'constant',
                         constant_values=((0, 0), (0, 0), (0, 0)))
            print("after pad y ",img.shape)

    elif array.shape[1]>256:
        dy2 = round(array.shape[1] - 256)
        if dy2 % 2 ==0:
            print("before crop y ",array.shape)
            img = array[:,round((dy2/2)):round(array.shape[1]-(dy2/2)),:]
            print("after crop y ",img.shape)
            if img.shape[1]>256:
                img = img [::-1,:]
        else:
            print("before crop y ",array.shape)
            img = array[:,round((dy2/2)):round(array.shape[1]-(dy2/2)+0.5),:]
            print("after crop y ",img.shape)

    if array.shape[2]<256:
        dx1 = round(256-array.shape[2])
        if dx1 % 2 == 0:
            print("before pad x ",array.shape)
            img = np.pad(img,((0,0),(0,0),(round(dx1/2),round(dx1/2))),'constant', constant_values=((0, 0),(0, 0),(0, 0)))
            print("after pad x ",img.shape)
        else:
            print("before pad x ",array.shape)
            img = np.pad(img,((0,0),(0,0),(round((dx1/2)),round((dx1/2)+0.5))),'constant', constant_values=((0, 0),(0, 0),(0, 0)))
            print("after pad x ",img.shape)
    elif array.shape[2]>256:
        dx2 = round(array.shape[2]-256)
        if dx2 % 2 == 0:
            print("before crop x ",array.shape)
            img = img[:,:,round((dx2/2)):round(array.shape[2]-dx2/2)]
            print("after crop x ",img.shape)
            if img.shape[2]>256:
                img = img [::,:-1]
        else:
            print("before crop x ",array.shape)
            img = img[:,:,round((dx2/2)):round(array.shape[2]-(dx2/2)+0.5)]
            print("after crop x ",img.shape)

    if img.shape[2]>256:
        di = img.shape[2]-256
        img = img [:,:,:-di]

    if img.shape[1]>256:
        dii = img.shape[1]-256
        img = img [:,:-dii,:]


    return img


