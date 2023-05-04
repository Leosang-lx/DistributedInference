import time
import numpy as np
import torch.nn as nn
import torch
from models.googlenet import Inception

# layer = nn.Conv2d(3, 64, 3, 1, 1)
input_shape = (1, 256, 28, 28)
x = torch.randn(input_shape)
times = []
layer = Inception(256, 128, 128, 192, 32, 96, 64)
layers = [layer.branch1, layer.branch2, layer.branch3, layer.branch4]
for i in range(4):
    branch = layers[i]
    print(i)
    # xx = x[..., :i]
    ts = []
    for _ in range(3):
        start = time.time()
        _ = branch(x)
        end = time.time()
        consumption = end - start
        ts.append(consumption)
    times.append(ts)

print(times)
a = np.array(times)
print(a.mean(axis=-1))

# data = [[0.07609844207763672, 0.0747520923614502, 0.07428288459777832, 0.0742650032043457, 0.07422614097595215], [0.13095879554748535, 0.13023805618286133, 0.15701532363891602, 0.14210724830627441, 0.1304304599761963], [0.18929409980773926, 0.18883395195007324, 0.1881258487701416, 0.18814587593078613, 0.18807697296142578], [0.2439110279083252, 0.24348688125610352, 0.24247169494628906, 0.2426285743713379, 0.24268293380737305], [0.30405640602111816, 0.30324411392211914, 0.3024723529815674, 0.3028101921081543, 0.3025624752044678], [0.3579139709472656, 0.3568742275238037, 0.3572986125946045, 0.3565995693206787, 0.3575022220611572], [0.4179213047027588, 0.41949987411499023, 0.41916704177856445, 0.417478084564209, 0.41805267333984375], [0.475691556930542, 0.4750676155090332, 0.4739220142364502, 0.4746696949005127, 0.4735391139984131], [0.5442245006561279, 0.5422263145446777, 0.5408506393432617, 0.5410184860229492, 0.5410170555114746], [0.606088399887085, 0.6005680561065674, 0.5991747379302979, 0.5994486808776855, 0.6029574871063232], [0.6732981204986572, 0.6740121841430664, 0.670668363571167, 0.6711258888244629, 0.6733877658843994], [0.7495689392089844, 0.7447495460510254, 0.7410058975219727, 0.7429664134979248, 0.7454521656036377], [0.8236782550811768, 0.8159520626068115, 0.8135886192321777, 0.8157050609588623, 0.8167848587036133], [0.8952813148498535, 0.8907430171966553, 0.894047737121582, 0.8934669494628906, 0.8897888660430908], [0.9784009456634521, 0.976313591003418, 0.9782662391662598, 0.9799535274505615, 0.979119062423706], [1.0647642612457275, 1.051806926727295, 1.0512166023254395, 1.0518689155578613, 1.0518672466278076], [1.2253391742706299, 1.2178096771240234, 1.2195754051208496, 1.2246997356414795, 1.2174146175384521], [1.534231185913086, 1.4515376091003418, 1.5869979858398438, 1.5511863231658936, 1.4659695625305176], [1.7622339725494385, 1.7800946235656738, 1.782033920288086, 1.7734761238098145, 1.7901365756988525], [2.1134867668151855, 2.1102728843688965, 2.205162525177002, 2.1210994720458984, 2.133310317993164], [2.534640073776245, 2.4624178409576416, 2.525092363357544, 2.5377542972564697, 2.469843864440918], [2.9267780780792236, 2.8388476371765137, 2.9090704917907715, 2.85446834564209, 2.8374545574188232], [3.140575885772705, 3.096757650375366, 3.096112012863159, 3.132361650466919, 3.098706007003784], [3.327700138092041, 3.348482847213745, 3.2966277599334717, 3.268960475921631, 3.304582118988037], [3.5517730712890625, 3.54484486579895, 3.544192314147949, 3.5358269214630127, 3.5432424545288086], [3.7517893314361572, 3.744439125061035, 3.7500407695770264, 3.7410545349121094, 3.750070571899414], [3.084625482559204, 2.9494717121124268, 2.9296233654022217, 2.96122145652771, 2.9660303592681885], [3.05448579788208, 3.0492868423461914, 3.037736177444458, 3.0390172004699707, 3.0432941913604736], [3.328761339187622, 3.3442153930664062, 3.3510875701904297, 3.336092710494995, 3.3381736278533936], [3.4685425758361816, 3.4626693725585938, 3.4611446857452393, 3.4644877910614014, 3.457164764404297], [3.419088840484619, 3.3976786136627197, 3.389263868331909, 3.3923022747039795, 3.386702060699463], [3.579510450363159, 3.5537281036376953, 3.5632522106170654, 3.555172920227051, 3.5411789417266846], [3.831702709197998, 3.822686195373535, 3.828874111175537, 3.8194966316223145, 3.8233587741851807], [3.938638687133789, 3.916292428970337, 3.9331870079040527, 3.921098470687866, 3.936939001083374], [4.049975633621216, 4.033452987670898, 4.04146671295166, 4.035825729370117, 4.048164129257202], [4.2291038036346436, 4.225183486938477, 4.231914281845093, 4.237611770629883, 4.238211631774902], [4.44619607925415, 4.277787685394287, 4.241482496261597, 4.229975938796997, 4.2940146923065186], [4.526413440704346, 4.54061222076416, 4.5450828075408936, 4.539832353591919, 4.545192718505859], [4.692138195037842, 4.594458818435669, 4.528161287307739, 4.451850175857544, 4.539902925491333], [4.843138694763184, 4.574042558670044, 4.570397138595581, 4.477391719818115, 4.510634899139404], [5.015866279602051, 4.97074818611145, 4.8644118309021, 4.9542319774627686, 4.984380006790161], [5.114205360412598, 5.108560085296631, 5.10929536819458, 5.111217498779297, 5.10938024520874], [5.220266580581665, 5.233055591583252, 5.247658729553223, 5.236420154571533, 5.2427239418029785], [5.378960847854614, 5.378965854644775, 5.378085136413574, 5.364026308059692, 5.367020130157471], [5.512772798538208, 5.537973403930664, 5.532739877700806, 5.543319940567017, 5.522847414016724], [5.466073274612427, 5.431242942810059, 5.35056471824646, 5.443532943725586, 5.313879013061523], [5.770222425460815, 5.758857250213623, 5.7550225257873535, 5.77238655090332, 5.761429071426392], [6.148991107940674, 7.35287880897522, 7.3093085289001465, 7.336586236953735, 7.345254898071289], [5.977917194366455, 5.956360340118408, 5.895830392837524, 5.900141716003418, 5.849706649780273], [6.113316297531128, 6.0860209465026855, 6.090644836425781, 6.086350440979004, 6.092486143112183], [6.223849534988403, 5.848611116409302, 5.738041162490845, 5.768290996551514, 5.824243545532227], [6.345772743225098, 6.320624113082886, 6.3167946338653564, 6.313949108123779, 6.329719066619873], [6.511363506317139, 6.499468564987183, 6.498071670532227, 6.49418830871582, 6.504769802093506], [6.166879177093506, 6.03201150894165, 6.0461907386779785, 6.032595634460449, 6.0332324504852295], [6.787121772766113, 6.669339895248413, 6.68210768699646, 6.667190074920654, 6.70104718208313], [6.520672798156738, 6.232185363769531, 6.3616015911102295, 6.2204060554504395, 6.405593395233154]]
# print(data[26])
# a = np.array(data)
# a = a.mean(axis=-1)
# import matplotlib.pyplot as plt
# plt.plot(list(range(1, 57)), a)
# plt.show()



