#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <iomanip>
#include "utils.h"
#include "weight_loader.h"
#include "data_loader.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

memory conv2d_onednn_wo_bias(memory &INPUT,  std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights,
	tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti = 0);

memory bn_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var,
	tensor_dims &t_dims, float eps = 1.e-5f, int Acti = 0);

memory pooling_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int KH, int KW, int SH, int SW, int DH, int DW, int TP, int BP, int LP, int RP, int mode, int Acti = 0);

memory gap_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int Acti = 0);

memory fc_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias, 
	tensor_dims &t_dims, int OC, int Acti = 0);

memory eltwise_onednn(memory &INPUT, memory &INPUT2, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream);

memory activation_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	int mode = 0);

// imagenet label name 1000
static std::vector<std::string> class_names{  // 1000 classes
	"tench Tinca tinca","goldfish Carassius auratus","great white shark white shark man-eater man-eating shark Carcharodon carcharias","tiger shark Galeocerdo cuvieri","hammerhead hammerhead shark","electric ray crampfish numbfish torpedo","stingray","cock","hen","ostrich Struthio camelus","brambling Fringilla montifringilla","goldfinch Carduelis carduelis","house finch linnet Carpodacus mexicanus","junco snowbird","indigo bunting indigo finch indigo bird Passerina cyanea","robin American robin Turdus migratorius","bulbul","jay","magpie","chickadee","water ouzel dipper","kite","bald eagle American eagle Haliaeetus leucocephalus","vulture","great grey owl great gray owl Strix nebulosa","European fire salamander Salamandra salamandra","common newt Triturus vulgaris","eft","spotted salamander Ambystoma maculatum","axolotl mud puppy Ambystoma mexicanum","bullfrog Rana catesbeiana","tree frog tree-frog","tailed frog bell toad ribbed toad tailed toad Ascaphus trui","loggerhead loggerhead turtle Caretta caretta","leatherback turtle leatherback leathery turtle Dermochelys coriacea","mud turtle","terrapin","box turtle box tortoise","banded gecko","common iguana iguana Iguana iguana","American chameleon anole Anolis carolinensis",
	"whiptail whiptail lizard","agama","frilled lizard Chlamydosaurus kingi","alligator lizard","Gila monster Heloderma suspectum","green lizard Lacerta viridis","African chameleon Chamaeleo chamaeleon","Komodo dragon Komodo lizard dragon lizard giant lizard Varanus komodoensis","African crocodile Nile crocodile Crocodylus niloticus","American alligator Alligator mississipiensis","triceratops","thunder snake worm snake Carphophis amoenus","ringneck snake ring-necked snake ring snake","hognose snake puff adder sand viper","green snake grass snake","king snake kingsnake","garter snake grass snake","water snake","vine snake","night snake Hypsiglena torquata","boa constrictor Constrictor constrictor","rock python rock snake Python sebae","Indian cobra Naja naja","green mamba","sea snake","horned viper cerastes sand viper horned asp Cerastes cornutus","diamondback diamondback rattlesnake Crotalus adamanteus","sidewinder horned rattlesnake Crotalus cerastes","trilobite","harvestman daddy longlegs Phalangium opilio","scorpion","black and gold garden spider Argiope aurantia","barn spider Araneus cavaticus","garden spider Aranea diademata","black widow Latrodectus mactans","tarantula","wolf spider hunting spider","tick","centipede",
	"black grouse","ptarmigan","ruffed grouse partridge Bonasa umbellus","prairie chicken prairie grouse prairie fowl","peacock","quail","partridge","African grey African gray Psittacus erithacus","macaw","sulphur-crested cockatoo Kakatoe galerita Cacatua galerita","lorikeet","coucal","bee eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted merganser Mergus serrator","goose","black swan Cygnus atratus","tusker","echidna spiny anteater anteater","platypus duckbill duckbilled platypus duck-billed platypus Ornithorhynchus anatinus","wallaby brush kangaroo","koala koala bear kangaroo bear native bear Phascolarctos cinereus","wombat","jellyfish","sea anemone anemone","brain coral","flatworm platyhelminth","nematode nematode worm roundworm","conch","snail","slug","sea slug nudibranch","chiton coat-of-mail shell sea cradle polyplacophore","chambered nautilus pearly nautilus nautilus","Dungeness crab Cancer magister","rock crab Cancer irroratus","fiddler crab","king crab Alaska crab Alaskan king crab Alaska king crab Paralithodes camtschatica","American lobster Northern lobster Maine lobster Homarus americanus","spiny lobster langouste rock lobster crawfish crayfish sea crawfish","crayfish crawfish crawdad crawdaddy",
	"hermit crab","isopod","white stork Ciconia ciconia","black stork Ciconia nigra","spoonbill","flamingo","little blue heron Egretta caerulea","American egret great white heron Egretta albus","bittern","crane","limpkin Aramus pictus","European gallinule Porphyrio porphyrio","American coot marsh hen mud hen water hen Fulica americana","bustard","ruddy turnstone Arenaria interpres","red-backed sandpiper dunlin Erolia alpina","redshank Tringa totanus","dowitcher","oystercatcher oyster catcher","pelican","king penguin Aptenodytes patagonica","albatross mollymawk","grey whale gray whale devilfish Eschrichtius gibbosus Eschrichtius robustus","killer whale killer orca grampus sea wolf Orcinus orca","dugong Dugong dugon","sea lion","Chihuahua","Japanese spaniel","Maltese dog Maltese terrier Maltese","Pekinese Pekingese Peke","Shih-Tzu","Blenheim spaniel","papillon","toy terrier","Rhodesian ridgeback","Afghan hound Afghan","basset basset hound","beagle","bloodhound sleuthhound","bluetick","black-and-tan coonhound",
	"Walker hound Walker foxhound","English foxhound","redbone","borzoi Russian wolfhound","Irish wolfhound","Italian greyhound","whippet","Ibizan hound Ibizan Podenco","Norwegian elkhound elkhound","otterhound otter hound","Saluki gazelle hound","Scottish deerhound deerhound","Weimaraner","Staffordshire bullterrier Staffordshire bull terrier","American Staffordshire terrier Staffordshire terrier American pit bull terrier pit bull terrier","Bedlington terrier","Border terrier","Kerry blue terrier","Irish terrier","Norfolk terrier","Norwich terrier","Yorkshire terrier","wire-haired fox terrier","Lakeland terrier","Sealyham terrier Sealyham","Airedale Airedale terrier","cairn cairn terrier","Australian terrier","Dandie Dinmont Dandie Dinmont terrier","Boston bull Boston terrier","miniature schnauzer","giant schnauzer","standard schnauzer","Scotch terrier Scottish terrier Scottie","Tibetan terrier chrysanthemum dog","silky terrier Sydney silky","soft-coated wheaten terrier","West Highland white terrier","Lhasa Lhasa apso","flat-coated retriever","curly-coated retriever","golden retriever","Labrador retriever","Chesapeake Bay retriever","German short-haired pointer","vizsla Hungarian pointer","English setter",
	"Irish setter red setter","Gordon setter","Brittany spaniel","clumber clumber spaniel","English springer English springer spaniel","Welsh springer spaniel","cocker spaniel English cocker spaniel cocker","Sussex spaniel","Irish water spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old English sheepdog bobtail","Shetland sheepdog Shetland sheep dog Shetland","collie","Border collie","Bouvier des Flandres Bouviers des Flandres","Rottweiler","German shepherd German shepherd dog German police dog alsatian","Doberman Doberman pinscher","miniature pinscher","Greater Swiss Mountain dog","Bernese mountain dog","Appenzeller","EntleBucher","boxer","bull mastiff","Tibetan mastiff","French bulldog","Great Dane","Saint Bernard St Bernard","Eskimo dog husky","malamute malemute Alaskan malamute","Siberian husky","dalmatian coach dog carriage dog","affenpinscher monkey pinscher monkey dog","basenji","pug pug-dog","Leonberg","Newfoundland Newfoundland dog","Great Pyrenees","Samoyed Samoyede","Pomeranian","chow chow chow","keeshond","Brabancon griffon","Pembroke Pembroke Welsh corgi","Cardigan Cardigan Welsh corgi","toy poodle","miniature poodle","standard poodle","Mexican hairless",
	"timber wolf grey wolf gray wolf Canis lupus","white wolf Arctic wolf Canis lupus tundrarum","red wolf maned wolf Canis rufus Canis niger","coyote prairie wolf brush wolf Canis latrans","dingo warrigal warragal Canis dingo","dhole Cuon alpinus","African hunting dog hyena dog Cape hunting dog Lycaon pictus","hyena hyaena","red fox Vulpes vulpes","kit fox Vulpes macrotis","Arctic fox white fox Alopex lagopus","grey fox gray fox Urocyon cinereoargenteus","tabby tabby cat","tiger cat","Persian cat","Siamese cat Siamese","Egyptian cat","cougar puma catamount mountain lion painter panther Felis concolor","lynx catamount","leopard Panthera pardus","snow leopard ounce Panthera uncia","jaguar panther Panthera onca Felis onca","lion king of beasts Panthera leo","tiger Panthera tigris","cheetah chetah Acinonyx jubatus","brown bear bruin Ursus arctos","American black bear black bear Ursus americanus Euarctos americanus","ice bear polar bear Ursus Maritimus Thalarctos maritimus","sloth bear Melursus ursinus Ursus ursinus","mongoose","meerkat mierkat","tiger beetle","ladybug ladybeetle lady beetle ladybird ladybird beetle","ground beetle carabid beetle","long-horned beetle longicorn longicorn beetle","leaf beetle chrysomelid",
	"dung beetle","rhinoceros beetle","weevil","fly","bee","ant emmet pismire","grasshopper hopper","cricket","walking stick walkingstick stick insect","cockroach roach","mantis mantid","cicada cicala","leafhopper","lacewing lacewing fly","dragonfly darning needle devils darning needle sewing needle snake feeder snake doctor mosquito hawk skeeter hawk","damselfly","admiral","ringlet ringlet butterfly","monarch monarch butterfly milkweed butterfly Danaus plexippus","cabbage butterfly","sulphur butterfly sulfur butterfly","lycaenid lycaenid butterfly","starfish sea star","sea urchin","sea cucumber holothurian","wood rabbit cottontail cottontail rabbit","hare","Angora Angora rabbit","hamster","porcupine hedgehog","fox squirrel eastern fox squirrel Sciurus niger","marmot","beaver","guinea pig Cavia cobaya","sorrel","zebra","hog pig grunter squealer Sus scrofa","wild boar boar Sus scrofa","warthog","hippopotamus hippo river horse Hippopotamus amphibius","ox","water buffalo water ox Asiatic buffalo Bubalus bubalis","bison","ram tup","bighorn bighorn sheep cimarron Rocky Mountain bighorn Rocky Mountain sheep Ovis canadensis","ibex Capra ibex","hartebeest","impala Aepyceros melampus","gazelle","Arabian camel dromedary Camelus dromedarius",
	"llama","weasel","mink","polecat fitch foulmart foumart Mustela putorius","black-footed ferret ferret Mustela nigripes","otter","skunk polecat wood pussy","badger","armadillo","three-toed sloth ai Bradypus tridactylus","orangutan orang orangutang Pongo pygmaeus","gorilla Gorilla gorilla","chimpanzee chimp Pan troglodytes","gibbon Hylobates lar","siamang Hylobates syndactylus Symphalangus syndactylus","guenon guenon monkey","patas hussar monkey Erythrocebus patas","baboon","macaque","langur","colobus colobus monkey","proboscis monkey Nasalis larvatus","marmoset","capuchin ringtail Cebus capucinus","howler monkey howler","titi titi monkey","spider monkey Ateles geoffroyi","squirrel monkey Saimiri sciureus","Madagascar cat ring-tailed lemur Lemur catta","indri indris Indri indri Indri brevicaudatus","Indian elephant Elephas maximus","African elephant Loxodonta africana","lesser panda red panda panda bear cat cat bear Ailurus fulgens","giant panda panda panda bear coon bear Ailuropoda melanoleuca","barracouta snoek","eel",
	"coho cohoe coho salmon blue jack silver salmon Oncorhynchus kisutch","rock beauty Holocanthus tricolor","anemone fish","sturgeon","gar garfish garpike billfish Lepisosteus osseus","lionfish","puffer pufferfish blowfish globefish","abacus","abaya","academic gown academic robe judges robe","accordion piano accordion squeeze box","acoustic guitar","aircraft carrier carrier flattop attack aircraft carrier","airliner","airship dirigible","altar","ambulance","amphibian amphibious vehicle","analog clock","apiary bee house","apron","ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin","assault rifle assault gun","backpack back pack knapsack packsack rucksack haversack","bakery bakeshop bakehouse","balance beam beam","balloon","ballpoint ballpoint pen ballpen Biro","Band Aid","banjo","bannister banister balustrade balusters handrail","barbell","barber chair","barbershop","barn","barometer","barrel cask","barrow garden cart lawn cart wheelbarrow","baseball","basketball","bassinet","bassoon","bathing cap swimming cap","bath towel","bathtub bathing tub bath tub","beach wagon station wagon wagon estate car beach waggon station waggon waggon","beacon lighthouse beacon light pharos","beaker",
	"bearskin busby shako","beer bottle","beer glass","bell cote bell cot","bib","bicycle-built-for-two tandem bicycle tandem","bikini two-piece","binder ring-binder","binoculars field glasses opera glasses","birdhouse","boathouse","bobsled bobsleigh bob","bolo tie bolo bola tie bola","bonnet poke bonnet","bookcase","bookshop bookstore bookstall","bottlecap","bow","bow tie bow-tie bowtie","brass memorial tablet plaque","brassiere bra bandeau","breakwater groin groyne mole bulwark seawall jetty","breastplate aegis egis","broom","bucket pail","buckle","bulletproof vest","bullet train bullet","butcher shop meat market","cab hack taxi taxicab","caldron cauldron","candle taper wax light","cannon","canoe","can opener tin opener","cardigan","car mirror","carousel carrousel merry-go-round roundabout whirligig","carpenters kit tool kit","carton","car wheel","cash machine cash dispenser automated teller machine automatic teller machine automated teller automatic teller ATM","cassette","cassette player","castle","catamaran","CD player","cello violoncello","cellular telephone cellular phone cellphone cell mobile phone","chain","chainlink fence","chain mail ring mail mail chain armor chain armour ring armor ring armour","chain saw chainsaw",
	"chest","chiffonier commode","chime bell gong","china cabinet china closet","Christmas stocking","church church building","cinema movie theater movie theatre movie house picture palace","cleaver meat cleaver chopper","cliff dwelling","cloak","clog geta patten sabot","cocktail shaker","coffee mug","coffeepot","coil spiral volute whorl helix","combination lock","computer keyboard keypad","confectionery confectionary candy store","container ship containership container vessel","convertible","corkscrew bottle screw","cornet horn trumpet trump","cowboy boot","cowboy hat ten-gallon hat","cradle","crane","crash helmet","crate","crib cot","Crock Pot","croquet ball","crutch","cuirass","dam dike dyke","desk","desktop computer","dial telephone dial phone","diaper nappy napkin","digital clock","digital watch","dining table board","dishrag dishcloth","dishwasher dish washer dishwashing machine","disk brake disc brake","dock dockage docking facility","dogsled dog sled dog sleigh","dome","doormat welcome mat","drilling platform offshore rig","drum membranophone tympan","drumstick","dumbbell","Dutch oven","electric fan blower","electric guitar","electric locomotive","entertainment center","envelope","espresso maker","face powder",
	"feather boa boa","file file cabinet filing cabinet","fireboat","fire engine fire truck","fire screen fireguard","flagpole flagstaff","flute transverse flute","folding chair","football helmet","forklift","fountain","fountain pen","four-poster","freight car","French horn horn","frying pan frypan skillet","fur coat","garbage truck dustcart","gasmask respirator gas helmet","gas pump gasoline pump petrol pump island dispenser","goblet","go-kart","golf ball","golfcart golf cart","gondola","gong tam-tam","gown","grand piano grand","greenhouse nursery glasshouse","grille radiator grille","grocery store grocery food market market","guillotine","hair slide","hair spray","half track","hammer","hamper","hand blower blow dryer blow drier hair dryer hair drier","hand-held computer hand-held microcomputer","handkerchief hankie hanky hankey","hard disc hard disk fixed disk","harmonica mouth organ harp mouth harp","harp","harvester reaper","hatchet","holster","home theater home theatre","honeycomb","hook claw","hoopskirt crinoline",
	"horizontal bar high bar","horse cart horse-cart","hourglass","iPod","iron smoothing iron","jack-o-lantern","jean blue jean denim","jeep landrover","jersey T-shirt tee shirt","jigsaw puzzle","jinrikisha ricksha rickshaw","joystick","kimono","knee pad","knot","lab coat laboratory coat","ladle","lampshade lamp shade","laptop laptop computer","lawn mower mower","lens cap lens cover","letter opener paper knife paperknife","library","lifeboat","lighter light igniter ignitor","limousine limo","liner ocean liner","lipstick lip rouge","Loafer","lotion","loudspeaker speaker speaker unit loudspeaker system speaker system","loupe jewelers loupe","lumbermill sawmill","magnetic compass","mailbag postbag","mailbox letter box","maillot","maillot tank suit","manhole cover","maraca","marimba xylophone","mask","matchstick","maypole","maze labyrinth","measuring cup","medicine chest medicine cabinet","megalith megalithic structure","microphone mike","microwave microwave oven","military uniform","milk can","minibus","miniskirt mini","minivan","missile","mitten","mixing bowl","mobile home manufactured home","Model T","modem","monastery","monitor","moped","mortar","mortarboard","mosque","mosquito net","motor scooter scooter",
	"mountain bike all-terrain bike off-roader","mountain tent","mouse computer mouse","mousetrap","moving van","muzzle","nail","neck brace","necklace","nipple","notebook notebook computer","obelisk","oboe hautboy hautbois","ocarina sweet potato","odometer hodometer mileometer milometer","oil filter","organ pipe organ","oscilloscope scope cathode-ray oscilloscope CRO","overskirt","oxcart","oxygen mask","packet","paddle boat paddle","paddlewheel paddle wheel","padlock","paintbrush","pajama pyjama pjs jammies","palace","panpipe pandean pipe syrinx","paper towel","parachute chute","parallel bars bars","park bench","parking meter","passenger car coach carriage","patio terrace","pay-phone pay-station","pedestal plinth footstall","pencil box pencil case","pencil sharpener","perfume essence","Petri dish","photocopier","pick plectrum plectron","pickelhaube","picket fence paling","pickup pickup truck","pier","piggy bank penny bank","pill bottle","pillow","ping-pong ball","pinwheel","pirate pirate ship","pitcher ewer","plane carpenters plane woodworking plane","planetarium","plastic bag","plate rack","plow plough","plunger plumbers helper","Polaroid camera Polaroid Land camera","pole","police van police wagon paddy wagon patrol wagon wagon black Maria",
	"poncho","pool table billiard table snooker table","pop bottle soda bottle","pot flowerpot","potters wheel","power drill","prayer rug prayer mat","printer","prison prison house","projectile missile","projector","puck hockey puck","punching bag punch bag punching ball punchball","purse","quill quill pen","quilt comforter comfort puff","racer race car racing car","racket racquet","radiator","radio wireless","radio telescope radio reflector","rain barrel","recreational vehicle RV R.V.","reel","reflex camera","refrigerator icebox","remote control remote","restaurant eating house eating place eatery","revolver six-gun six-shooter","rifle","rocking chair rocker","rotisserie","rubber eraser rubber pencil eraser","rugby ball","rule ruler","running shoe","safe","safety pin","saltshaker salt shaker","sandal","sarong","sax saxophone","scabbard","scale weighing machine","school bus","schooner","scoreboard","screen CRT screen","screw","screwdriver","seat belt seatbelt","sewing machine","shield buckler","shoe shop shoe-shop shoe store","shoji","shopping basket","shopping cart","shovel","shower cap","shower curtain","ski","ski mask","sleeping bag","slide rule slipstick","sliding door","slot one-armed bandit","snorkel","snowmobile","snowplow snowplough",
	"soap dispenser","soccer ball","sock","solar dish solar collector solar furnace","sombrero","soup bowl","space bar","space heater","space shuttle","spatula","speedboat","spider web spiders web","spindle","sports car sport car","spotlight spot","stage","steam locomotive","steel arch bridge","steel drum","stethoscope","stole","stone wall","stopwatch stop watch","stove","strainer","streetcar tram tramcar trolley trolley car","stretcher","studio couch day bed","stupa tope","submarine pigboat sub U-boat","suit suit of clothes","sundial","sunglass","sunglasses dark glasses shades","sunscreen sunblock sun blocker","suspension bridge","swab swob mop","sweatshirt","swimming trunks bathing trunks","swing","switch electric switch electrical switch","syringe","table lamp","tank army tank armored combat vehicle armoured combat vehicle","tape player","teapot","teddy teddy bear","television television system","tennis ball","thatch thatched roof","theater curtain theatre curtain","thimble","thresher thrasher threshing machine","throne","tile roof","toaster","tobacco shop tobacconist shop tobacconist","toilet seat","torch","totem pole","tow truck tow car wrecker","toyshop","tractor","trailer truck tractor trailer trucking rig rig articulated lorry semi",
	"tray","trench coat","tricycle trike velocipede","trimaran","tripod","triumphal arch","trolleybus trolley coach trackless trolley","trombone","tub vat","turnstile","typewriter keyboard","umbrella","unicycle monocycle","upright upright piano","vacuum vacuum cleaner","vase","vault","velvet","vending machine","vestment","viaduct","violin fiddle","volleyball","waffle iron","wall clock","wallet billfold notecase pocketbook","wardrobe closet press","warplane military plane","washbasin handbasin washbowl lavabo wash-hand basin","washer automatic washer washing machine","water bottle","water jug","water tower","whiskey jug","whistle","wig","window screen","window shade","Windsor tie","wine bottle","wing","wok","wooden spoon","wool woolen woollen","worm fence snake fence snake-rail fence Virginia fence","wreck","yawl","yurt","web site website internet site site","comic book","crossword puzzle crossword","street sign","traffic light traffic signal stoplight",	"book jacket dust cover dust jacket dust wrapper","menu","plate","guacamole","consomme","hot pot hotpot","trifle","ice cream icecream","ice lolly lolly lollipop popsicle","French loaf","bagel beigel","pretzel","cheeseburger","hotdog hot dog red hot","mashed potato","head cabbage","broccoli",
	"cauliflower","zucchini courgette","spaghetti squash","acorn squash","butternut squash","cucumber cuke","artichoke globe artichoke","bell pepper","cardoon","mushroom","Granny Smith","strawberry","orange","lemon","fig","pineapple ananas","banana","jackfruit jak jack","custard apple","pomegranate","hay","carbonara","chocolate sauce chocolate syrup","dough","meat loaf meatloaf","pizza pizza pie","potpie","burrito","red wine","espresso","cup","eggnog","alp","bubble","cliff drop drop-off","coral reef","geyser","lakeside lakeshore","promontory headland head foreland","sandbar sand bar","seashore coast seacoast sea-coast","valley vale","volcano","ballplayer baseball player","groom bridegroom","scuba diver","rapeseed","daisy","yellow ladys slipper yellow lady-slipper Cypripedium calceolus Cypripedium parviflorum","corn","acorn","hip rose hip rosehip","buckeye horse chestnut conker","coral fungus","agaric",
	"gyromitra","stinkhorn carrion fungus","earthstar","hen-of-the-woods hen of the woods Polyporus frondosus Grifola frondosa","bolete","ear spike capitulum","toilet tissue toilet paper bathroom tissue"
};

void resnet18(std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &stream, std::map<std::string, Weights> &weightMap,
	int batch_size,int input_channel,int input_width,int input_height, tensor_dims &t_dims) {

	std::vector<float> inputs(batch_size * input_channel * input_height * input_width);
	//[inputs]
	auto inputs_src_md = memory::desc({ batch_size, input_channel, input_height, input_width}, dt::f32, tag::nchw);
	auto inputs_src_md_memory = memory(inputs_src_md, engine);
	write_to_dnnl_memory(inputs.data(), inputs_src_md_memory);

	// net work
	memory conv1 = conv2d_onednn_wo_bias(inputs_src_md_memory, net, net_args, engine, stream, weightMap["conv1.weight"].values,  t_dims, 64, 7, 7, 2, 2, 3, 3, 3, 3, 0);
	memory bn_relu1 = bn_onednn(conv1, net, net_args, engine, stream, weightMap["bn1.weight"].values, weightMap["bn1.bias"].values, weightMap["bn1.running_mean"].values, weightMap["bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory pool1 = pooling_onednn(bn_relu1, net, net_args, engine, stream, t_dims, 3, 3, 2, 2, 0, 0, 1, 1, 1, 1, 1, 0); 

	// layer1 
	// basicBlock1
	memory layer1_conv1_1 = conv2d_onednn_wo_bias(pool1, net, net_args, engine, stream, weightMap["layer1.0.conv1.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer1_bn_relu1_1 = bn_onednn(layer1_conv1_1, net, net_args, engine, stream, weightMap["layer1.0.bn1.weight"].values, weightMap["layer1.0.bn1.bias"].values, weightMap["layer1.0.bn1.running_mean"].values, weightMap["layer1.0.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer1_conv1_2 = conv2d_onednn_wo_bias(layer1_bn_relu1_1, net, net_args, engine, stream, weightMap["layer1.0.conv2.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer1_bn1_2 = bn_onednn(layer1_conv1_2, net, net_args, engine, stream, weightMap["layer1.0.bn2.weight"].values, weightMap["layer1.0.bn2.bias"].values, weightMap["layer1.0.bn2.running_mean"].values, weightMap["layer1.0.bn2.running_var"].values, t_dims, 1.e-5f, 0);
	
	memory layer1_elt_sum1_3 = eltwise_onednn(pool1, layer1_bn1_2, net, net_args, engine, stream);
	memory layer1_relu1_3 = activation_onednn(layer1_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer1_conv1_4 = conv2d_onednn_wo_bias(layer1_relu1_3, net, net_args, engine, stream, weightMap["layer1.1.conv1.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer1_bn_relu1_4 = bn_onednn(layer1_conv1_4, net, net_args, engine, stream, weightMap["layer1.1.bn1.weight"].values, weightMap["layer1.1.bn1.bias"].values, weightMap["layer1.1.bn1.running_mean"].values, weightMap["layer1.1.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer1_conv1_5 = conv2d_onednn_wo_bias(layer1_bn_relu1_4, net, net_args, engine, stream, weightMap["layer1.1.conv2.weight"].values, t_dims, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer1_bn5 = bn_onednn(layer1_conv1_5, net, net_args, engine, stream, weightMap["layer1.1.bn2.weight"].values, weightMap["layer1.1.bn2.bias"].values, weightMap["layer1.1.bn2.running_mean"].values, weightMap["layer1.1.bn2.running_var"].values, t_dims, 1.e-5f, 0);
	
	memory layer1_elt_sum1_5 = eltwise_onednn(layer1_relu1_3, layer1_bn5, net, net_args, engine, stream);
	memory layer1_relu1_5 = activation_onednn(layer1_elt_sum1_5, net, net_args, engine, stream);
	// layer1 

	// layer2 
	// basicBlock1
	tensor_dims t_dims2 = t_dims;
	memory layer2_conv1_1 = conv2d_onednn_wo_bias(layer1_relu1_5, net, net_args, engine, stream, weightMap["layer2.0.conv1.weight"].values, t_dims, 128, 3, 3, 2, 2, 1, 1, 1, 1, 0);
	memory layer2_bn_relu1_1 = bn_onednn(layer2_conv1_1, net, net_args, engine, stream, weightMap["layer2.0.bn1.weight"].values, weightMap["layer2.0.bn1.bias"].values, weightMap["layer2.0.bn1.running_mean"].values, weightMap["layer2.0.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer2_conv1_2 = conv2d_onednn_wo_bias(layer2_bn_relu1_1, net, net_args, engine, stream, weightMap["layer2.0.conv2.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer2_bn1_2 = bn_onednn(layer2_conv1_2, net, net_args, engine, stream, weightMap["layer2.0.bn2.weight"].values, weightMap["layer2.0.bn2.bias"].values, weightMap["layer2.0.bn2.running_mean"].values, weightMap["layer2.0.bn2.running_var"].values, t_dims, 1.e-5f, 0);
	
	memory layer2_down_conv1_2 = conv2d_onednn_wo_bias(layer1_relu1_5, net, net_args, engine, stream, weightMap["layer2.0.downsample.0.weight"].values, t_dims2, 128, 1, 1, 2, 2, 0, 0, 0, 0, 0);
	memory layer2_down_bn1_2 = bn_onednn(layer2_down_conv1_2, net, net_args, engine, stream, weightMap["layer2.0.downsample.1.weight"].values, weightMap["layer2.0.downsample.1.bias"].values, weightMap["layer2.0.downsample.1.running_mean"].values, weightMap["layer2.0.downsample.1.running_var"].values, t_dims, 1.e-5f, 0);
	
	memory layer2_elt_sum1_3 = eltwise_onednn(layer2_down_bn1_2, layer2_bn1_2, net, net_args, engine, stream);
	memory layer2_relu1_3 = activation_onednn(layer2_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer2_conv1_4 = conv2d_onednn_wo_bias(layer2_relu1_3, net, net_args, engine, stream, weightMap["layer2.1.conv1.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer2_bn_relu1_4 = bn_onednn(layer2_conv1_4, net, net_args, engine, stream, weightMap["layer2.1.bn1.weight"].values, weightMap["layer2.1.bn1.bias"].values, weightMap["layer2.1.bn1.running_mean"].values, weightMap["layer2.1.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer2_conv1_5 = conv2d_onednn_wo_bias(layer2_bn_relu1_4, net, net_args, engine, stream, weightMap["layer2.1.conv2.weight"].values, t_dims, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer2_bn5 = bn_onednn(layer2_conv1_5, net, net_args, engine, stream, weightMap["layer2.1.bn2.weight"].values, weightMap["layer2.1.bn2.bias"].values, weightMap["layer2.1.bn2.running_mean"].values, weightMap["layer2.1.bn2.running_var"].values, t_dims, 1.e-5f, 0);
	
	memory layer2_elt_sum1_5 = eltwise_onednn(layer2_relu1_3, layer2_bn5, net, net_args, engine, stream);
	memory layer2_relu1_5 = activation_onednn(layer2_elt_sum1_5, net, net_args, engine, stream);
	// layer2 

	// layer3 
	// basicBlock1
	t_dims2 = t_dims;
	memory layer3_conv1_1 = conv2d_onednn_wo_bias(layer2_relu1_5, net, net_args, engine, stream, weightMap["layer3.0.conv1.weight"].values, t_dims, 256, 3, 3, 2, 2, 1, 1, 1, 1, 0);
	memory layer3_bn_relu1_1 = bn_onednn(layer3_conv1_1, net, net_args, engine, stream, weightMap["layer3.0.bn1.weight"].values, weightMap["layer3.0.bn1.bias"].values, weightMap["layer3.0.bn1.running_mean"].values, weightMap["layer3.0.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer3_conv1_2 = conv2d_onednn_wo_bias(layer3_bn_relu1_1, net, net_args, engine, stream, weightMap["layer3.0.conv2.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer3_bn1_2 = bn_onednn(layer3_conv1_2, net, net_args, engine, stream, weightMap["layer3.0.bn2.weight"].values, weightMap["layer3.0.bn2.bias"].values, weightMap["layer3.0.bn2.running_mean"].values, weightMap["layer3.0.bn2.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer3_down_conv1_2 = conv2d_onednn_wo_bias(layer2_relu1_5, net, net_args, engine, stream, weightMap["layer3.0.downsample.0.weight"].values, t_dims2, 256, 1, 1, 2, 2, 0, 0, 0, 0, 0);
	memory layer3_down_bn1_2 = bn_onednn(layer3_down_conv1_2, net, net_args, engine, stream, weightMap["layer3.0.downsample.1.weight"].values, weightMap["layer3.0.downsample.1.bias"].values, weightMap["layer3.0.downsample.1.running_mean"].values, weightMap["layer3.0.downsample.1.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer3_elt_sum1_3 = eltwise_onednn(layer3_down_bn1_2, layer3_bn1_2, net, net_args, engine, stream);
	memory layer3_relu1_3 = activation_onednn(layer3_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer3_conv1_4 = conv2d_onednn_wo_bias(layer3_relu1_3, net, net_args, engine, stream, weightMap["layer3.1.conv1.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer3_bn_relu1_4 = bn_onednn(layer3_conv1_4, net, net_args, engine, stream, weightMap["layer3.1.bn1.weight"].values, weightMap["layer3.1.bn1.bias"].values, weightMap["layer3.1.bn1.running_mean"].values, weightMap["layer3.1.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer3_conv1_5 = conv2d_onednn_wo_bias(layer3_bn_relu1_4, net, net_args, engine, stream, weightMap["layer3.1.conv2.weight"].values, t_dims, 256, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer3_bn5 = bn_onednn(layer3_conv1_5, net, net_args, engine, stream, weightMap["layer3.1.bn2.weight"].values, weightMap["layer3.1.bn2.bias"].values, weightMap["layer3.1.bn2.running_mean"].values, weightMap["layer3.1.bn2.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer3_elt_sum1_5 = eltwise_onednn(layer3_relu1_3, layer3_bn5, net, net_args, engine, stream);
	memory layer3_relu1_5 = activation_onednn(layer3_elt_sum1_5, net, net_args, engine, stream);
	// layer3 

	// layer4 
	// basicBlock1
	t_dims2 = t_dims;
	memory layer4_conv1_1 = conv2d_onednn_wo_bias(layer3_relu1_5, net, net_args, engine, stream, weightMap["layer4.0.conv1.weight"].values, t_dims, 512, 3, 3, 2, 2, 1, 1, 1, 1, 0);
	memory layer4_bn_relu1_1 = bn_onednn(layer4_conv1_1, net, net_args, engine, stream, weightMap["layer4.0.bn1.weight"].values, weightMap["layer4.0.bn1.bias"].values, weightMap["layer4.0.bn1.running_mean"].values, weightMap["layer4.0.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer4_conv1_2 = conv2d_onednn_wo_bias(layer4_bn_relu1_1, net, net_args, engine, stream, weightMap["layer4.0.conv2.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer4_bn1_2 = bn_onednn(layer4_conv1_2, net, net_args, engine, stream, weightMap["layer4.0.bn2.weight"].values, weightMap["layer4.0.bn2.bias"].values, weightMap["layer4.0.bn2.running_mean"].values, weightMap["layer4.0.bn2.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer4_down_conv1_2 = conv2d_onednn_wo_bias(layer3_relu1_5, net, net_args, engine, stream, weightMap["layer4.0.downsample.0.weight"].values, t_dims2, 512, 1, 1, 2, 2, 0, 0, 0, 0, 0);
	memory layer4_down_bn1_2 = bn_onednn(layer4_down_conv1_2, net, net_args, engine, stream, weightMap["layer4.0.downsample.1.weight"].values, weightMap["layer4.0.downsample.1.bias"].values, weightMap["layer4.0.downsample.1.running_mean"].values, weightMap["layer4.0.downsample.1.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer4_elt_sum1_3 = eltwise_onednn(layer4_down_bn1_2, layer4_bn1_2, net, net_args, engine, stream);
	memory layer4_relu1_3 = activation_onednn(layer4_elt_sum1_3, net, net_args, engine, stream);
	// basicBlock2
	memory layer4_conv1_4 = conv2d_onednn_wo_bias(layer4_relu1_3, net, net_args, engine, stream, weightMap["layer4.1.conv1.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer4_bn_relu1_4 = bn_onednn(layer4_conv1_4, net, net_args, engine, stream, weightMap["layer4.1.bn1.weight"].values, weightMap["layer4.1.bn1.bias"].values, weightMap["layer4.1.bn1.running_mean"].values, weightMap["layer4.1.bn1.running_var"].values, t_dims, 1.e-5f, 1);
	memory layer4_conv1_5 = conv2d_onednn_wo_bias(layer4_bn_relu1_4, net, net_args, engine, stream, weightMap["layer4.1.conv2.weight"].values, t_dims, 512, 3, 3, 1, 1, 1, 1, 1, 1, 0);
	memory layer4_bn5 = bn_onednn(layer4_conv1_5, net, net_args, engine, stream, weightMap["layer4.1.bn2.weight"].values, weightMap["layer4.1.bn2.bias"].values, weightMap["layer4.1.bn2.running_mean"].values, weightMap["layer4.1.bn2.running_var"].values, t_dims, 1.e-5f, 0);

	memory layer4_elt_sum1_5 = eltwise_onednn(layer4_relu1_3, layer4_bn5, net, net_args, engine, stream);
	memory layer4_relu1_5 = activation_onednn(layer4_elt_sum1_5, net, net_args, engine, stream);
	// layer4 

	memory global_avg_pooling = gap_onednn(layer4_relu1_5, net, net_args, engine, stream, t_dims);
	memory fc1 = fc_onednn(global_avg_pooling, net, net_args, engine, stream, weightMap["fc.weight"].values, weightMap["fc.bias"].values, t_dims, 1000);

	
}

int main(int argc, char **argv) {
	std::cout << "igpu count: "<<  dnnl::engine::get_count(dnnl::engine::kind::gpu) << std::endl;

	// Weight load =============================================================
	std::string file = "../model/resnet18.wts";
	std::map<std::string, Weights> weightMap = loadWeights(file);
	std::cout << "weight load done!" << std::endl;

	// ONEDNN =============================================================
	//[Initialize engine and stream]
	int batch_size = 1;
	int input_channel = 3;
	int input_width = 224;
	int input_height = 224;
	tensor_dims t_dims{ batch_size , input_channel, input_height, input_width };
	engine engine(engine::kind::cpu, 0);
	stream stream(engine);
	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;
	resnet18(net, net_args, engine, stream, weightMap, batch_size, input_channel, input_height, input_width, t_dims); //100th dur time : 9078 ms - > 7019 ms
	assert(net.size() == net_args.size() && "something is missing");

	// Image load =============================================================
	std::string img_dir = "../data";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) {
		std::cerr << "data search error" << std::endl;
		exit(0);
	}
	// Image preprocess ============================================================
	cv::Mat img(input_height, input_width, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(batch_size * input_height * input_width * input_channel);
	std::vector<float> inputs(input.size());
	for (int idx = 0; idx < batch_size; idx++) {
		cv::Mat ori_img = cv::imread(file_names[idx]);
		cv::resize(ori_img, img, img.size(), cv::INTER_LINEAR);
		int offset = idx * input_height * input_width * input_channel;
		memcpy(input.data() + offset, img.data, input_height * input_width * input_channel);
	}
	preprocess(inputs, input, batch_size, input_channel, input_height, input_width);
	//tofile(inputs);

	//[Execute model]
	uint64_t dur_time = 0;
	uint64_t iter_count = 100;
	write_to_dnnl_memory(inputs.data(), net_args.at(0).find(DNNL_ARG_SRC)->second);

	for (int j = 0; j < iter_count; ++j) {

		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		write_to_dnnl_memory(inputs.data(), net_args.at(0).find(DNNL_ARG_SRC)->second);
		stream.wait();

		for (size_t i = 0; i < net.size(); ++i) {
			net.at(i).execute(stream, net_args.at(i));	
		}

		stream.wait();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
		dur_time += dur;
	}

	std::vector<float> outputs(t_dims.N * t_dims.IC* t_dims.IH* t_dims.IW);
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	//tofile(outputs);
	//valueCheck(outputs, t_dims.N , t_dims.IC, t_dims.IH, t_dims.IW);
	// 6) ��� ���
	std::cout << "==================================================" << std::endl;
	std::cout << "===============" << " resnet18 " << "===============" << std::endl;
	std::cout << iter_count << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
	int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << ", Class Name : " << class_names[max_index] << std::endl;
	std::cout << "==================================================" << std::endl;
	std::cout << "layer count : " << net.size() << std::endl;

	return 0;
}


memory conv2d_onednn_wo_bias(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	auto conv_weights_md = memory::desc({ OC, t_dims.IC, KH, KW }, dt::f32, tag::oihw);
	auto user_weights_mem = memory(conv_weights_md, engine);
	write_to_dnnl_memory(weights.data(), user_weights_mem);

	auto conv_dst_md = memory::desc({ t_dims.N, OC, OH, OW }, dt::f32, tag::nchw);
	memory OUTPUT = memory(conv_dst_md, engine);

	auto conv_src_md2 = memory::desc({ t_dims.N, t_dims.IC, t_dims.IH, t_dims.IW }, dt::f32, tag::any);
	auto conv_weights_md2 = memory::desc({ OC, t_dims.IC, KH, KW }, dt::f32, tag::any);
	auto conv_dst_md2 = memory::desc({ t_dims.N, OC, OH, OW }, dt::f32, tag::any);

	// Create operation descriptor.
	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_auto, 
		conv_src_md2, conv_weights_md2, conv_dst_md2, { SH, SW }, { TP, LP }, { BP, RP });

	// Activation func
	convolution_forward::primitive_desc conv_pd;
	if (Acti == 1) {
		// Create primitive post-ops (ReLU).
		const float scale = 1.f;
		const float alpha = 0.f;
		const float beta = 0.f;
		post_ops conv_ops;
		conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
		primitive_attr conv_attr;
		conv_attr.set_post_ops(conv_ops);
		conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
	}
	else { // linear
		conv_pd = convolution_forward::primitive_desc(conv_desc, engine);
	}

	auto conv_src_mem = INPUT;
	auto conv_weights_mem = user_weights_mem;
	auto conv_dst_mem = OUTPUT;

	if (conv_pd.src_desc() != INPUT.get_desc()) {
		conv_src_mem = memory(conv_pd.src_desc(), engine);
		reorder(INPUT, conv_src_mem).execute(engine_stream, INPUT, conv_src_mem);
	}

	if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
		conv_weights_mem = memory(conv_pd.weights_desc(), engine);
		reorder(user_weights_mem, conv_weights_mem).execute(engine_stream, user_weights_mem, conv_weights_mem);
	}

	if (conv_pd.dst_desc() != OUTPUT.get_desc()) {
		conv_dst_mem = memory(conv_pd.dst_desc(), engine);
	}

	// Create the primitive.
	net.push_back(convolution_forward(conv_pd));

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, user_weights_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;
	return OUTPUT;
}

memory bn_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var, tensor_dims &t_dims, float eps, int Acti)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	std::vector<float> scale_shift(2 * t_dims.IC);
	memcpy(scale_shift.data(), scale.data(), sizeof(float) * t_dims.IC);
	memcpy(scale_shift.data()+ t_dims.IC, shift.data(), sizeof(float) * t_dims.IC);

	auto scale_shift_mem_md = memory::desc({ 2, t_dims.IC }, dt::f32, tag::nc);
	auto scale_shift_mem = memory(scale_shift_mem_md, engine);
	write_to_dnnl_memory(scale_shift.data(), scale_shift_mem);

	auto mean_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto mean_mem = memory(mean_mem_md, engine);
	write_to_dnnl_memory(mean.data(), mean_mem);

	auto variance_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto variance_mem = memory(variance_mem_md, engine);
	write_to_dnnl_memory(var.data(), variance_mem);

	// Create primitive descriptor.
	batch_normalization_forward::primitive_desc bnorm_pd;

	if (Acti == 1) { // relu
		auto bnorm_d = batch_normalization_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), eps,
			normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu);
		bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
	}
	else { // linear
		auto bnorm_d = batch_normalization_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), 1.e-5f,
			normalization_flags::use_global_stats | normalization_flags::use_scale_shift);
		bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
	}

	// Create the primitive.
	//auto bnorm_prim = batch_normalization_forward(bnorm_pd);

	net.push_back(batch_normalization_forward(bnorm_pd));
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_MEAN, mean_mem },
		{ DNNL_ARG_VARIANCE, variance_mem },
		{ DNNL_ARG_SCALE_SHIFT, scale_shift_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});
	return OUTPUT;
}

memory pooling_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int KH, int KW, int SH, int SW, int DH, int DW, int TP, int BP, int LP, int RP, int mode, int Acti)
{
	//const memory::dim OH = (t_dims.IH + (TP + BP) - DH * (KH - 1) - 1) / SH + 1; // dilation = 1 pytorch 
	//const memory::dim OW = (t_dims.IW + (LP + RP) - DW * (KW - 1) - 1) / SW + 1;

	const memory::dim OH = (t_dims.IH + (TP + BP) - (DH * (KH - 1) + KH)) / SH + 1; // dilation = 0 oneDNN
	const memory::dim OW = (t_dims.IW + (LP + RP) - (DW * (KW - 1) + KW)) / SW + 1;

	auto pooling_dst_md = memory::desc({ t_dims.N, t_dims.IC, OH, OW }, dt::f32, tag::nchw);
	memory OUTPUT = memory(pooling_dst_md, engine);

	pooling_v2_forward::primitive_desc pooling_pd;

	if (mode == 1) {//pooling_max
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_max,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}
	else {//pooling_avg
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}

	// Create the primitive.
	//auto pooling_prim = pooling_v2_forward(pooling_pd);

	net.push_back(pooling_v2_forward(pooling_pd));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IH = OH;
	t_dims.IW = OW;

	return OUTPUT;
}

memory gap_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int Acti)
{
	auto gap_dst_md = memory::desc({ t_dims.N, t_dims.IC ,1, 1}, dt::f32, tag::nchw);
	memory OUTPUT = memory(gap_dst_md, engine);

	auto gap_d = reduction::desc(algorithm::reduction_mean, INPUT.get_desc(), gap_dst_md, 0.f, 0.f);
	reduction::primitive_desc gap_pd = reduction::primitive_desc(gap_d, engine);

	// Create the primitive.
	//auto gap_prim = reduction(gap_pd);

	net.push_back(reduction(gap_pd));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});
	t_dims.IH = 1;
	t_dims.IW = 1;
	return OUTPUT;
}

memory fc_onednn(memory &INPUT,  std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias, tensor_dims &t_dims, int OC, int Acti)
{
	auto fc_dst_md = memory::desc({ t_dims.N, OC }, dt::f32, tag::nc);
	memory OUTPUT = memory(fc_dst_md, engine);

	auto fc_weights_md = memory::desc({ OC, t_dims.IC,1,1 }, dt::f32, tag::oihw);
	auto fc_weights_mem = memory(fc_weights_md, engine);
	write_to_dnnl_memory(weights.data(), fc_weights_mem);

	auto fc_bias_md = memory::desc({ OC }, dt::f32, tag::a);
	auto fc_bias_mem = memory(fc_bias_md, engine);
	write_to_dnnl_memory(bias.data(), fc_bias_mem);

	// Create operation descriptor.
	auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), fc_weights_md, fc_bias_md, OUTPUT.get_desc());

	// Activation func
	inner_product_forward::primitive_desc fc_pd;

	if (Acti == 1) {
		// Create primitive post-ops (ReLU).
		const float scale = 1.f;
		const float alpha = 0.f;
		const float beta = 0.f;
		post_ops fc_ops;
		fc_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
		primitive_attr fc_attr;
		fc_attr.set_post_ops(fc_ops);
		fc_pd = inner_product_forward::primitive_desc(fc_desc, fc_attr, engine);
	}
	else { // linear
		fc_pd = inner_product_forward::primitive_desc(fc_desc, engine);
	}

	// Create the primitive.
	//auto fc_prim = inner_product_forward(fc_pd);
	net.push_back(inner_product_forward(fc_pd));

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, fc_weights_mem },
		{ DNNL_ARG_BIAS, fc_bias_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IC = OC;
	return OUTPUT;
}

memory eltwise_onednn(memory &INPUT, memory &INPUT2, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	// Create primitive descriptor.
	auto sum_pd = sum::primitive_desc({ 1, 1 }, { INPUT.get_desc() , INPUT2.get_desc() }, engine);

	// Create the primitive.
	//auto sum_prim = sum(sum_pd);

	net.push_back(sum(sum_pd));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_MULTIPLE_SRC + 0, INPUT },
		{ DNNL_ARG_MULTIPLE_SRC + 1, INPUT2 },
		{ DNNL_ARG_DST, OUTPUT }
		});

	return OUTPUT;
}

// activation
memory activation_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	int mode)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	eltwise_forward::primitive_desc eltwise_pd;

	if (mode == 0) {//relu
		auto eltwise_d = eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_relu, INPUT.get_desc(), 0.f, 0.f);
		eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, engine);
	}

	//auto eltwise_prim = eltwise_forward(eltwise_pd);

	net.push_back(eltwise_forward(eltwise_pd));
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	return OUTPUT;
}