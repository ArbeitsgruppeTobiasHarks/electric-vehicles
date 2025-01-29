from networkloading import *
from iteration import *
from animationdata import *

G = Network()
G.addNode("s5")
G.addNode("d2d6")
G.addNode("s4d3")
G.addNode("d4d5")
G.addNode("s1s2")
G.addNode("s3s6d1")

G.addEdge("s5","s5",1,0)

G.addEdge("s5","d2d6",ExtendedRational(1),ExtendedRational(1))
G.addEdge("d2d6","s4d3",ExtendedRational(1),ExtendedRational(1))
G.addEdge("s4d3","d4d5",ExtendedRational(1),ExtendedRational(1))
G.addEdge("s3s6d1","d2d6",ExtendedRational(3),ExtendedRational(1))
G.addEdge("s1s2","d2d6",ExtendedRational(72),ExtendedRational(1))
G.addEdge("s1s2","s3s6d1",ExtendedRational(9),ExtendedRational(1))
G.addEdge("d2d6","s3s6d1",ExtendedRational(72),ExtendedRational(4))

p1a = Path([G.edges[6]])
p1b = Path([G.edges[5],G.edges[7]])
p2a = Path([G.edges[5]])
p2b = Path([G.edges[6],G.edges[4]])
p3 = Path([G.edges[4],G.edges[2]])
p4 = Path([G.edges[3]])
p5 = Path([G.edges[1],G.edges[2],G.edges[3]])
p6 = Path([G.edges[4]])

M = 100
h = PartialFlowPathBased(G,7)
h.setPaths(0,[p1a],[PWConst([0,1],[0],0)])
h.setPaths(1,[p1b],[PWConst([0,1],[72],0)])
h.setPaths(2,[p2b],[PWConst([0,1],[9],0)])
h.setPaths(3,[p3],[PWConst([2,3],[2],0)])
h.setPaths(4,[p4],[PWConst([6,6+M],[1],0)])
h.setPaths(5,[p5],[PWConst([3,4],[1],0)])
h.setPaths(6,[p6],[PWConst([6,7],[3*M+9],0)])
#h.setPaths(0,[p1a],[PWConst([0,1],[0],0)])
#h.setPaths(1,[p1a,p1b],[PWConst([0,ExtendedRational(18,81)],[72],0),PWConst([ExtendedRational(18,81),1],[72],0)])
#h.setPaths(2,[p2b],[PWConst([0,1],[9],0)])
#h.setPaths(3,[p3],[PWConst([2,3],[2],0)])
#h.setPaths(4,[p4],[PWConst([6,6+M],[1],0)])
#h.setPaths(5,[p5],[PWConst([3,4],[1],0)])
#h.setPaths(6,[p6],[PWConst([6,7],[3*M+9],0)])

f = networkLoading(h,2*M)

#f.queues[G.edges[4]].drawGraph(0,M-1).show()

print(f)

generateCSV(f,0,10,41)

## Our standard example ##
# G = Network()
# G.addNode("s")
# G.addNode("u")
# G.addNode("v")
# G.addNode("t")
# G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(1))
# G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(2))
# G.addEdge("u", "v", ExtendedRational(1), ExtendedRational(1))
# G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(1))
# G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(2))
#
#
# p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
# p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
# p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
#
# times = [ExtendedRational(x) for x in [Fraction(0, 1), Fraction(1, 1), Fraction(7, 5), Fraction(79, 45), Fraction(533, 255), Fraction(60959, 25245), Fraction(87973, 32175), Fraction(645883231, 211679325), Fraction(20352428839, 6044620725), Fraction(102763256971871, 27908013887325), Fraction(38113179216969589, 9535238078169375), Fraction(252728456720421615199, 58613108466507148125), Fraction(370328783636493893629813, 80045968462426595289375), Fraction(9721281192644073548976735839, 1967449858837983285617548125), Fraction(369116333645531547520604449597, 70232387662440863038874784375), Fraction(5884175068934993254068485193929524831, 1056363306091530810182687504415628125), Fraction(10446391101941864475731853564876729086729, 1775150820290273197613917717356077446875), Fraction(56254848400700133856793020388953403224573871711, 9074299395248372174403112441713512428575628125), Fraction(5165072024104986294669764321404308931456585020561269, 792927404989128174552967970344329571862986011609375), Fraction(8516290197799605026078085037864710327099721726814143645279, 1247169348703035064524603072611574620707419219122002828125), Fraction(54627848924376713157711242516891239689502508044128592204519709, 7647678913187031572245700093975596965248675570346738944484375), Fraction(20452983984005701382459891002427076573600020927659196348595271268304479, 2742558324666403802703893523849986072521028627842873059616823122828125), Fraction(1753030505445120871449902478675615618204865152897781826806430171710845158693, 225551492038037662778430473069639526155006136841264738986590966827325859375), Fraction(780334062678853004098073836362886922942396285794809824351187500112712575343076312671, 96495227079759265096427450485578023548840613602655113003123961045966764644178828125), Fraction(4533697632046113227474128485594996397643242695063341986510264003139032886700158307262355317,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              539640454727132499427763087184432623836669690941632202337644124122577128407771995419609375), Fraction(473464856597617158264243384159222732196706270921580240486913004779499163850976283431468041414903391, 54321988446693302202514244546409609358625208263495113931942769422144094127472946298750768978828125), Fraction(98860175530477131986318780907914295226180771995537220742860581153709386163760630411528933392439109603419, 10947408375978079621359552846372338465918065125981315402457557059690078148191816048994339411550644140625), Fraction(4572420573369255577988074626279621330446636410240954350771793596412002405052274292127605347180716741285646166132319, 489288984789565548186499672140028525385850578227342579566941251037360071049982725158430398901283433738934098828125), Fraction(32531174468153043288756991271987757733709388502080452242842442729927686867630149169505117162527732444552247356348485464841, 3367756724027001968176886172339967312948382084619768083118224020881166019124476478718170808506623498006153679790023046875), Fraction(703317969722471456069003366506823523580946937592568289609004914444010067693950629891697551608344207732512119349472490344012058986079, 70513974460420357749008951168808110310017429854063667796645691864131103942485392763281527834463065832950878821799993537206098828125), Fraction(15274489640839601790969789082765778387145816080506027841592261649330027815626978704928696269838505869709445905351265386035302899291893335333, 1484584384806571474442622807241802028217243743398270712277726942364309482087626679637013738936562160834663171793595585208389842291005859375), Fraction(1724034282525046518085671722332060292694199854363468971559341772486668424046303335977435213661231896970852137031677171026013801196102824676322523219551, 162594155288574649724883047228538680889469945498562240524092109936320120711900670666569061237285716321891915046310450629833775640818812921321298828125)]]
#
# lambdas = [ExtendedRational(x)
#            for x in [2+1.0/2**i for i in range(len(times)-1)]]
#
# times = [ExtendedRational(0, 1), ExtendedRational(1, 1), ExtendedRational(
#     7, 5), ExtendedRational(7, 5)+ExtendedRational(1, 1000), ExtendedRational(100, 1)]
# lambdas = [ExtendedRational(3, 1), ExtendedRational(
#     5, 2), 2+ExtendedRational(2, 9), 2+ExtendedRational(2, 9)]
#
# f1 = PWConst(times, [l for l in lambdas], ExtendedRational(0))
# f2 = PWConst(times, [ExtendedRational(3-l)
#                      for l in lambdas], ExtendedRational(0))
#
# pathInflowRates = PartialFlowPathBased(G,1)
# pathInflowRates.setPaths(0, [p1, p2], [f1, f2])
#
# flow = networkLoading(pathInflowRates)
# #flow = networkLoading(pathInflowRates, ExtendedRational(50), verbose=True)
#
#
# print(flow)
#
# for theta in times[:-1]:
#     print("Starting at ", theta, " along path P1: ", flow.pathArrivalTime(
#         p1, theta), " ≈ ", float(flow.pathArrivalTime(p1, theta)))
#     print("Starting at ", theta, " along path P2: ", flow.pathArrivalTime(
#         p2, theta), " ≈ ", float(flow.pathArrivalTime(p2, theta)))
#     print("Starting at ", theta, " along path P3: ", flow.pathArrivalTime(
#         p3, theta), " ≈ ", float(flow.pathArrivalTime(p3, theta)))
#
# theta = ExtendedRational(7, 5) + ExtendedRational(1, 100)
# print("Arrival time at u over e_1 = ",
#       flow.pathArrivalTime(Path([G.edges[0]]), theta))
# print("Arrival time at v over e_1 = ", flow.pathArrivalTime(
#     Path([G.edges[2]]), flow.pathArrivalTime(Path([G.edges[0]]), theta)))
# print("Arrival time at u over e_2 = ",
#       flow.pathArrivalTime(Path([G.edges[1]]), theta))
# print("Arrival time at v over e_2 = ", flow.pathArrivalTime(
#     Path([G.edges[2]]), flow.pathArrivalTime(Path([G.edges[1]]), theta)))
#
# # Check feasibility
# print(flow.checkFeasibility(10))

# flow.fPlus[G.edges[0], 0].drawGraph(0, 3).show()
# flow.fPlus[G.edges[1], 1].drawGraph(0, 3).show()
#flow.queues[G.edges[0]].drawGraph(0, 5).show()
#flow.queues[G.edges[2]].drawGraph(0, 5).show()
#flow.fMinus[(G.edges[2],0)].drawGraph(0, 10).show()

# G = Network()
# G.addNode("s")
# G.addNode("a")
# G.addNode("b")
# G.addNode("u")
# G.addNode("v")
# G.addNode("c")
# G.addNode("d")
# G.addNode("t")
# G.addEdge("s", "a", ExtendedRational(2), ExtendedRational(1,2))
# G.addEdge("a", "u", ExtendedRational(3), ExtendedRational(1,2))
# G.addEdge("s", "b", ExtendedRational(2), ExtendedRational(1))
# G.addEdge("b", "u", ExtendedRational(3), ExtendedRational(1))
# G.addEdge("u", "v", ExtendedRational(1), ExtendedRational(1))
# G.addEdge("v", "c", ExtendedRational(2), ExtendedRational(1,2))
# G.addEdge("c", "t", ExtendedRational(3), ExtendedRational(1,2))
# G.addEdge("v", "d", ExtendedRational(2), ExtendedRational(1))
# G.addEdge("d", "t", ExtendedRational(3), ExtendedRational(1))

# G = Network()
# G.addNode("s")
# G.addNode("u")
# G.addNode("v")
# G.addNode("t")
# G.addEdge("s","t",1,3)
# G.addEdge("s","u",2,1)
# G.addEdge("u","v",2,1)
# G.addEdge("v","t",1,1)
# G.addEdge("v","s",1,1)

# f = fixedPointIteration(G,1,[(G.getNode("s"),G.getNode("t"),PWConst([0,10],[10],0))],50,100,False)
# print(f)
# networkLoading(f).fPlus[G.edges[0],0].drawGraph(0,10).show()
# networkLoading(f).fPlus[G.edges[2],1].drawGraph(0,10).show()
# networkLoading(f).queues[G.edges[0]].drawGraph(0,10).show()
# networkLoading(f).queues[G.edges[4]].drawGraph(0,10).show()


## An  adjusted example ##

# G = Network()
# G.addNode("s")
# G.addNode("u")
# G.addNode("v")
# G.addNode("t")
# G.addEdge("s", "u", ExtendedRational(3), ExtendedRational(1))
# G.addEdge("s", "u", ExtendedRational(3), ExtendedRational(2))
# G.addEdge("u", "v", ExtendedRational(2), ExtendedRational(1))
# G.addEdge("v", "t", ExtendedRational(1), ExtendedRational(1))
# G.addEdge("v", "t", ExtendedRational(1), ExtendedRational(2))
#
#
# p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
# p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
# p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
#
# times = [ExtendedRational(x) for x in [0, Fraction(1, 3), 20]]
# lambdas = [ExtendedRational(x) for x in [Fraction(3, 2), Fraction(3, 2)]]
#
# f1 = PWConst(times, [l for l in lambdas], ExtendedRational(0))
# f2 = PWConst(times, [ExtendedRational(3-l)
#                      for l in lambdas], ExtendedRational(0))
#
# pathInflowRates = PartialFlowPathBased(G,1)
# pathInflowRates.setPaths(0, [p1, p2], [f1, f2])
#
# flow = networkLoading(pathInflowRates, ExtendedRational(50), verbose=True)
#
#
# print(flow)
#
# for theta in [0, Fraction(1, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     print("Starting at ", theta, " along path P1: ", flow.pathArrivalTime(
#         p1, theta), " ≈ ", float(flow.pathArrivalTime(p1, theta)))
#     print("Starting at ", theta, " along path P2: ", flow.pathArrivalTime(
#         p2, theta), " ≈ ", float(flow.pathArrivalTime(p2, theta)))
#     print("Starting at ", theta, " along path P3: ", flow.pathArrivalTime(
#         p3, theta), " ≈ ", float(flow.pathArrivalTime(p3, theta)))
