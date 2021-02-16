from .PF_Pascal import PF_Pascal_dataset

def load(name, root_path, mode):
	if name == 'PF_Pascal':
		ds_obj = PF_Pascal_dataset(root_path)
		if mode == "classification":
			train_ds, val_ds = ds_obj.load_classification()
			return train_ds, val_ds
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError
