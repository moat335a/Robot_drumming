import itertools
from utils import get_indexes
import os
import numpy as np



class Get_hits:
    def __init__(self,data) -> None:
        """data can be the path to folder containing files in pre defined file structure 
        or a dictionary object containig nessecary files"""
        if isinstance(data,str):
            self.data=self.get_files(data)
        else:
            self.data = data
        self.test_d()
        self.indexes=get_indexes(self.data["hit_times"],self.data["times"])
        grouped =self.MSeifert()
        i=0
        assert np.sum([i+d["duration"] for d in grouped ]) == self.data["joint_positions"].shape[0], "shapes don't match "
        self.get_hit(groups=grouped)
            
    def MSeifert(self):
        return [{'value': k, 'duration': len(list(v))} for k, v in 
                itertools.groupby(self.data["joint_positions"].tolist())]

    def get_hit(self,groups):
        finished =[]
        for n in range(len(self.indexes)):
            for t,dat in enumerate(groups[self.indexes[n]:]):
                if dat["duration"] != 1:
                    if t<300:
                        continue
                    finished.append(t)
        self.get_min_hits(np.min(finished))

    def get_min_hits(self,mini):
        self.hits = np.zeros((len(self.indexes),mini*2,4))
        for n in range(len(self.indexes)):
            self.hits[n] = self.data["joint_positions"][self.indexes[n]-mini:self.indexes[n]+mini]
    
    def test_d(self):
        """test for the data relative times to not accidentally be to close to each other"""
        t=0
        for _ in range(len(self.data["relativ_times"])//2):
            assert ((self.data["relativ_times"][t]>1)or (int(self.data["relativ_times"][t])==0)) ,self.data["relativ_times"][t] 

            t+=2

        
    @staticmethod
    def get_files(path):
        datas = {}
        for file in os.listdir(path):
            if not file.endswith("npy"):
                new_path =os.path.join(path,file)
                for f in os.listdir(new_path):
                    datas[f[:-4]] = np.load(os.path.join(new_path,f))
                for f in os.listdir(new_path):
                    if f!="times.npy":
                        datas[f[:-4]] = datas[f[:-4]][~np.isnan(datas["times"])]
                datas["times"] = datas["times"][~np.isnan(datas["times"])] 
            
            else:
                datas[file[:-4]] = np.load(os.path.join(path,file))
        return datas


