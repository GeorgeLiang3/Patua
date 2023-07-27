
# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')

import gempy as gp
from GemPhy.Geophysics.utils.util import constant64
from gempy.core.tensor.modeltf_var import ModelTF
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
# %%
class PutuaModel():
    def __init__(self) -> None:
        
        self.P = {}
        self.P['HypP'] = {}
        self.P['HypP']['jupyter'] = False
        self.P['HypP']['ErrorType'] = 'Global'
        self.P['DataTypes'] = ['Grav']

        self.P['xy_origin']=[318000,4379246, 1200-4000]
        self.P['xy_extent'] = [7000,9000,4000]
        # self.P['xy_extent'] = [9000,9400,4000]

        # Define the model limits
        self.P['xmin'] = self.P['xy_origin'][0]
        self.P['xmax'] = self.P['xy_origin'][0]+self.P['xy_extent'][0]

        self.P['ymin'] = self.P['xy_origin'][1]
        self.P['ymax'] = self.P['xy_origin'][1]+self.P['xy_extent'][1]

        self.P['zmin'] = self.P['xy_origin'][2]
        self.P['zmax'] = self.P['xy_origin'][2]+self.P['xy_extent'][2]

        # %%
        data_path = './Data/'
        self.geo_data = gp.create_data( extent=[self.P['xmin'], self.P['xmax'], self.P['ymin'], self.P['ymax'], self.P['zmin'], self.P['zmax']], resolution=[50, 50, 50],
                                path_o=data_path + "Patua_orientations.csv",
                                path_i=data_path + "Patua_surface_points.csv")


        gp.map_series_to_surfaces(self.geo_data, {"Intrusion": 'intrusion',
                                            "Fault_Series1": 'fault1',
                                            "Fault_Series2": 'fault2',
                                            "Fault_Series3": 'fault3',
                                            "Fault_Series4": 'fault4',
                                            "Fault_Series5": 'fault5',
                                            "Fault_Series6": 'fault6',
                                            "Fault_Series7": 'fault7',
                                            "Fault_Series8": 'fault8',
                                            "Fault_Series9": 'fault9',
                                            "Fault_Series10": 'fault10',
                                            "Fault_Series11": 'fault11',
                                            "Fault_Series12": 'fault12',
                                            "Sedimentary_Series": ('Volcanic_mafic','Volconic_felsic',
                                            'GT'),
                                            "Basement":'basement'
                                            }
                                            )

        # order_series = ['Fault_Series1',
        #                 'Fault_Series2',
        #                 'Fault_Series3',
        #                 'Fault_Series4',
        #                 'Fault_Series5',
        #                 'Fault_Series6',
        #                 'Fault_Series7',
        #                 'Fault_Series8',
        #                 'Fault_Series9',
        #                 'Fault_Series10',
        #                 'Fault_Series11',
        #                 'Fault_Series12',
        #                 'Sedimentary_Series',
        #                 'Basement']

        # self.geo_data.reorder_series(order_series)

        self.geo_data.set_is_fault(['Fault_Series1',
                            'Fault_Series2',
                            'Fault_Series3',
                            'Fault_Series4',
                            'Fault_Series5',
                            'Fault_Series6',
                            'Fault_Series7',
                            'Fault_Series8',
                            'Fault_Series9',
                            'Fault_Series10',
                            'Fault_Series11',
                            'Fault_Series12',
                            ])

        # Anisotropy
        mapping_object = {"Fault_Series1":np.array([1,1,1]),
                        "Fault_Series2":np.array([1,1,1]),
                        "Fault_Series3":np.array([1,1,1]),
                        "Fault_Series4":np.array([1,1,1]),
                        "Fault_Series5": np.array([1,1,1]),
                        "Fault_Series6":np.array([1,1,1]),
                        "Fault_Series7":np.array([1,1,1]),
                        "Fault_Series8": np.array([1,1,1]),
                        "Fault_Series9": np.array([1,1,1]),
                        "Fault_Series10":np.array([1,1,1]),
                        "Fault_Series11":np.array([1,1,1]),
                        "Fault_Series12":np.array([1,1,1]),
                        "Intrusion":np.array([1,1,0.1]),
                        "Sedimentary_Series": np.array([1,1,1]),
                        }
        gp.assign_global_anisotropy(self.geo_data,mapping_object)
        # assign densities to the model, fault series with -1 
        self.geo_data.add_surface_values([2.6, -1, -1, -1,-1, -1, -1,-1, -1, -1,-1, -1, -1, 2.1, 2.2, 2.3, 2.5], 'densities')

        # %%
        ## Initialize the model
    def init_model(self):
        model = ModelTF(self.geo_data)
        model.activate_regular_grid()
        model.create_tensorflow_graph(gradient = False)

        return model
    
def plot_results(mu,delta, max_slope,fix_points, path_name = None):

    densities = constant64(mu[-5:])
    sfp_z = tf.concat([fix_points[:,2],mu[:-5]],axis = -1)
    # concatenate the auxiliary densities, because faults has no densities but GemPy requires a value there
    auxiliary_densities = constant64([-1]*12)
    densities = tf.concat([densities[:1],auxiliary_densities,densities[1:]],axis = -1)

    sfp_xyz = concat_xy_and_scale(sfp_z,model_,model_.static_xy,all_points_shape)
    properties = tf.stack([model_prior.TFG.lith_label,densities],axis = 0)


    P = PutuaModel()
    model = ModelTF(P.geo_data)
    model.activate_regular_grid()
    model.geo_data.create_tensorflow_graph(delta = delta,gradient=True,max_slope = max_slope)
    model.compute_model(sfp_xyz)

    cross_section = gp.plot.plot_section(model, 
                                         cell_number=18,
                                         block = model.solutions.values_matrix,
                                         direction='y',
                                         show_grid=True, 
                                         show_data=True)
    if path_name is None:
        path_name = './Fig/model_nosmooth.png'
    cross_section.fig.savefig(path_name,dpi = 400)

# %%
if __name__ == "__main__":
    P = PutuaModel()
    # %%
    model = P.init_model()
    model.compute_model()
    # %%
    # gp._plot.plot_3d(model)


    # %%
    cross_section = gp.plot.plot_section(model, cell_number=18,
                            direction='y',show_grid=True, show_data=True)
    cross_section.fig.savefig('./Fig/model_nosmooth.png',dpi = 400)
