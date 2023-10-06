import pickle as pkl
import os


def save_data(data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
              file_name="gpmpc_d_runs", extension='.pkl', update_data=False):
    file_exists = False
    if os.path.exists(base_path+file_name+extension):
        file_exists = True
        if not update_data:
            file_name = file_name + '_1'
        else:
            with open(base_path+file_name+extension, 'rb') as f:
                old_data = pkl.load(f)
    with open(base_path+file_name+extension, 'wb') as f:
        if update_data and file_exists:
            if len(old_data) > len(data):
                data = old_data + data
        pkl.dump(data, f)


def read_data(base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
              file_name="gpmpc_d_runs", extension='.pkl'):
    assert os.path.exists(base_path+file_name+extension), "File to read desired trajectory to track does not exist"
    with open(base_path+file_name+extension, 'rb') as f:
        data = pkl.load(f)
    return data


def read_run_info(base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                  file_name="gpmpc_d_runs", extension='.pkl', print_data=True):
    with open(base_path+file_name+extension, 'rb') as f:
        data = pkl.load(f)
        for data_dict in data:
            if print_data:
                print(data_dict["average_run_time"])
                print(data_dict["collision_count"])
                print(data_dict["collision_idxs"])
                print(data_dict["cl_cost"])
    return data


def save_acc_n_loss(data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                    file_name="acc_n_loss_save", extension='.pkl'):
    if os.path.exists(base_path+file_name+extension):
        skip_read = False
    else:
        skip_read = True
    if not skip_read:
        with open(base_path+file_name+extension, 'rb') as f:
            base_data = pkl.load(f)
    else:
        base_data = {'acc': [], 'loss': []}
    with open(base_path+file_name+extension, 'wb') as f:
        base_data['acc'].append(data['acc'])
        base_data['loss'].append(data['loss'])
        pkl.dump(base_data, f)


def read_acc_n_loss(base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                    file_name="acc_n_loss_save", extension='.pkl', print_data=True):
    with open(base_path+file_name+extension, 'rb') as f:
        data = pkl.load(f)
    if print_data:
        print(data)
    return data
