import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of this script to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import numpy as np
from constants_AE import (
    EPOCHS,
    NUM_POINTS,
    LATENT_SIZE,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    SHAPE,
    AE_BEST_PTH
)
from utils.make_data import Data
from torch.utils import data
from autoencoder.pointnetAE import PCAutoEncoder
import torch
from gan.gan import Generator

import cv2
from constants_WGAN import (
    MU,
    SIGMA)
from utils.utils import noiseFunc
xml_head = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            
            <lookat origin="-1,3,-2" target="0,0,0" up="-1,0,0"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="512"/>
            <integer name="height" value="512"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="0.94, 0.82, 1.0"/>
    </bsdf>
    
"""

# "0.025"

xml_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            
            <translate x="{}" y="{}" z="{}"/>
            <rotate y="1" angle="-90"/>
            <rotate x="1" angle="-90"/>
            
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="30" y="30" z="1"/>
            <translate x="0" y="0" z="-1"/>
            <rotate y="1" angle="-120"/>
        </transform>

    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="20" y="20" z="1"/>
            <lookat origin="-10,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="10,9,10"/>
        </emitter>
    
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="23" y="23" z="1"/>
            <lookat origin="3,24,4" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="2,3,3"/>
        </emitter>
    </shape>
</scene>
"""

def standardize_bbox(pcl, points_per_object):
    if points_per_object > pcl.shape[0]:
        points_per_object = pcl.shape[0]
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result

def visualize_test_set(test_pointclouds, name):
    base_radius = 0.02
    coord_scale = 1.1   # changed be changed
    shift = 0.0125
    color = [0.06, 0.32, 0.73]
    for j,pointcloud in enumerate(test_pointclouds):
        pointcloud = pointcloud.detach().cpu().numpy()
        xml_segments = [xml_head]
        pointcloud = standardize_bbox(pointcloud, 1024)
        #pointcloud[:, 0] *= -1.
        pointcloud[:] *= coord_scale
        pointcloud[:, 2] += shift
        print('4:', pointcloud.shape, pointcloud.min(), pointcloud.max())
        
        for i in range(pointcloud.shape[0]):
            xml_segments.append(xml_ball_segment.format(base_radius * coord_scale, pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        with open('mitsuba_scene.xml', 'w') as f:
            f.write(xml_content)

        os.system("mitsuba mitsuba_scene.xml")

        hdr = cv2.imread('mitsuba_scene.exr', -1)
        # Simply clamp values to a 0-1 range for tone-mapping
        ldr = np.clip(hdr, 0, 1)
        # Color space conversion
        ldr = ldr**(1/2.2)
        # 0-255 remapping for bit-depth conversion
        ldr = 255.0 * ldr
        cv2.imwrite(f'{name}{j}.png', ldr)

if __name__ == '__main__':
    shape = "chair"
    pointcloud_set = Data(shape, NUM_POINTS)
    test_dataloader = data.DataLoader(pointcloud_set, batch_size=32)
    point_cloud_batch = next(iter(test_dataloader))
    point_cloud_batch = point_cloud_batch.to(DEVICE)

    visualize_test_set(point_cloud_batch, "original")

    autoencoder = PCAutoEncoder(LATENT_SIZE, NUM_POINTS)
    autoencoder.load_state_dict(torch.load(f"./weights/{shape}/autoencoder/best.pth"))
    autoencoder.to(DEVICE)
    autoencoder.eval()
    pointclouds = autoencoder(point_cloud_batch)
    visualize_test_set(pointclouds, "ae")

    fake_pointclouds = noiseFunc(MU, SIGMA, point_cloud_batch.shape[0], DEVICE, 128)
    generator = Generator(NUM_POINTS,128,0)
    generator.load_state_dict(torch.load(f"./weights/{shape}/generator/latent_best.pth"))
    generator.to(DEVICE)
    generator.eval()
    pointclouds = generator(fake_pointclouds)
    visualize_test_set(pointclouds, "latent_gan")

    fake_pointclouds = noiseFunc(MU, SIGMA, point_cloud_batch.shape[0], DEVICE, 128)
    generator = Generator(NUM_POINTS,128,0)
    generator.load_state_dict(torch.load(f"./weights/{shape}/generator/gan_best.pth"))
    generator.to(DEVICE)
    generator.eval()
    pointclouds = generator(fake_pointclouds)
    visualize_test_set(pointclouds, "wgan")

    fake_pointclouds = noiseFunc(MU, SIGMA, point_cloud_batch.shape[0], DEVICE, 128)
    encoder = autoencoder.encoder
    guided_latent = encoder(point_cloud_batch)[0]
    fake_pointclouds = torch.cat([fake_pointclouds, guided_latent], dim=1)
    generator = Generator(NUM_POINTS,128,16)
    generator.load_state_dict(torch.load(f"./weights/{shape}/generator/latent_guided_best.pth"))
    generator.to(DEVICE)
    generator.eval()
    pointclouds = generator(fake_pointclouds)
    visualize_test_set(pointclouds, "latent_guided_gan")

    fake_pointclouds = noiseFunc(MU, SIGMA, point_cloud_batch.shape[0], DEVICE, 144)
    encoder = autoencoder.encoder
    generator = Generator(NUM_POINTS,128,16)
    generator.load_state_dict(torch.load(f"./weights/{shape}/generator/latent_guided_best.pth"))
    generator.to(DEVICE)
    generator.eval()
    pointclouds = generator(fake_pointclouds)
    visualize_test_set(pointclouds, "latent_guided_pure_noise")
