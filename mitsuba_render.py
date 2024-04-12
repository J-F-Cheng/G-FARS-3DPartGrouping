# Import the library using the alias "mi"
import mitsuba as mi

# Set the variant of the renderer
mi.set_variant('scalar_rgb')
from mitsuba import ScalarTransform4f as T

def mitsuba_render(data_idx, gen_idx, save_dir, 
                   img_type="png",
                   sensor_width = 512,
                    sensor_height = 512,
                    sensor_sep = 25,            
                    phi = 225,  
                    radius = 2.5,
                    theta = -50,
                    spp = 512):
    # Load a scene
    scene = mi.load_file("{}/render_{}_{}.xml".format(save_dir, data_idx, gen_idx))

    

    def load_sensor(r, phi, theta):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': sensor_width,
                'height': sensor_height,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    sensor = load_sensor(radius, phi, theta)
    image = mi.render(scene, spp=spp, sensor=sensor)
    # Write the rendered image to an EXR file
    mi.util.write_bitmap("{}/render_{}_{}.{}".format(save_dir, data_idx, gen_idx, img_type), image)
