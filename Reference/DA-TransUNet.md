# DA-TransUNet: integrating spatial and channel dual attention with transformer U-net for medical image segmentation



## Page 1


TYPEOriginalResearch
PUBLISHED16May2024
DOI10.3389/fbioe.2024.1398237
DA-TransUNet: integrating spatial
and channel dual attention with
OPENACCESS
EDITEDBY transformer U-net for medical
AdrianElmi-Terander,
StockholmSpineCenter,Sweden
image segmentation
REVIEWEDBY
ShireenY.Elhabian,
TheUniversityofUtah,UnitedStates
NguyenQuocKhanhLe, Guanqun Sun1,2*, Yizhi Pan1, Weikun Kong3, Zichang Xu4,
TaipeiMedicalUniversity,Taiwan
Jianhua Ma5, Teeradaj Racharak2, Le-Minh Nguyen2* and
*CORRESPONDENCE
GuanqunSun, Junyi Xin1,6,7*
sun.guanqun@hmc.edu.cn
Le-MinhNguyen, 1SchoolofInformationEngineering,HangzhouMedicalCollege,Hangzhou,China,2Schoolof
nguyenml@jaist.ac.jp InformationScience,JapanAdvancedInstituteofScienceandTechnology,Nomi,Japan,3Department
JunyiXin, ofElectronicEngineering,TsinghuaUniversity,Beijing,China,4DepartmentofSystemsImmunology,
xinjunyi@hmc.edu.cn ImmunologyFrontierResearchInstitute(IFReC),OsakaUniversity,Suita,Japan,5FacultyofComputer
andInformationSciences,HoseiUniversity,Tokyo,Japan,6ZhejiangEngineeringResearchCenterfor
RECEIVED09March2024
BrainCognitionandBrainDiseasesDigitalMedicalInstruments,HangzhouMedicalCollege,Hangzhou,
ACCEPTED18April2024
China,7AcademyforAdvancedInterdisciplinaryStudiesofFutureHealth,HangzhouMedicalCollege,
PUBLISHED16May2024
Hangzhou,China
CITATION
SunG,PanY,KongW,XuZ,MaJ,RacharakT,
NguyenL-MandXinJ(2024),DA-TransUNet:
integratingspatialandchanneldualattention
withtransformerU-netformedical Accurate medical image segmentation is critical for disease quantification and
imagesegmentation. treatment evaluation. While traditional U-Net architectures and their
Front.Bioeng.Biotechnol.12:1398237.
transformer-integrated variants excel in automated segmentation tasks.
doi:10.3389/fbioe.2024.1398237
Existing models also struggle with parameter efficiency and computational
COPYRIGHT
complexity, often due to the extensive use of Transformers. However, they
©2024Sun,Pan,Kong,Xu,Ma,Racharak,
NguyenandXin.Thisisanopen-accessarticle lack the ability to harness the image’s intrinsic position and channel features.
distributedunderthetermsoftheCreative Research employing Dual Attention mechanisms of position and channel have
CommonsAttributionLicense(CCBY).Theuse,
notbeenspecificallyoptimizedforthehigh-detaildemandsofmedicalimages.
distributionorreproductioninotherforumsis
permitted,providedtheoriginalauthor(s)and To address these issues, this study proposes a novel deep medical image
thecopyrightowner(s)arecreditedandthatthe segmentation framework, called DA-TransUNet, aiming to integrate the
originalpublicationinthisjournaliscited,in
Transformer and dual attention block (DA-Block) into the traditional U-shaped
accordancewithacceptedacademicpractice.
Nouse,distributionorreproductionis architecture. Also, DA-TransUNet tailored for the high-detail requirements of
permittedwhichdoesnotcomplywiththese medicalimages,optimizestheintermittentchannelsofDualAttention(DA)and
terms.
employs DA in each skip-connection to effectively filter out irrelevant
information. This integration significantly enhances the model’s capability to
extract features, thereby improving the performance of medical image
segmentation. DA-TransUNet is validated in medical image segmentation
tasks, consistently outperforming state-of-the-art techniques across
5 datasets. In summary, DA-TransUNet has made significant strides in medical
image segmentation, offering new insights into existing techniques. It
strengthens model performance from the perspective of image features,
thereby advancing the development of high-precision automated medical
image diagnosis. The codes and parameters of our model will be publicly
available athttps://github.com/SUN-1024/DA-TransUnet.
KEYWORDS
U-net,medicalimagesegmentation,dualattention,transformer,deeplearning
FrontiersinBioengineeringandBiotechnology 01 frontiersin.org


## Page 2


Sunetal. 10.3389/fbioe.2024.1398237
1 Introduction
research opportunity in the task of medical image segmentation.
Therefore, addressing this gap and optimizing the integration of
Machinelearninganddeeplearningtechniqueshaveemergedas Transformers and Dual Attention mechanisms in the context of
powerful tools in biomedical research, revolutionizing disease medicalimagesegmentationposesasignificantchallengeforfuture
diagnosis, treatment planning, and personalized medicine (Le, research inthefield.
2024; Tran and Le, 2024). Medical image segmentation is the Toovercometheabovedrawbacks,recentstudieshaveexplored
process of delineating regions of interest within medical images the application of Transformer models in medical image
fordiagnosisandtreatmentplanning.Itservesasacornerstonein segmentation. Inspired by ViTs, TransUNet (Chen et al., 2021)
medicalimageanalysis.Manualsegmentationisbothaccurateand further combines thefunctionality of ViTs with the advantages of
affordableforpathologydiagnosisbutvitalinstandardizedclinical U-net in the field of medical image segmentation. Specifically, it
settings.Conversely,automatedsegmentationensuresareliableand employsatransformer’sencodertoprocesstheimageandemploys
consistent process,boosting efficiency, cutting down on laborand CNN and hopping connections for accurate up-sampling feature
costs,andpreservingaccuracy.Consequently,thereisasubstantial recovery, yet it neglects image-specific features like position and
demand for exceptionally accurate automated medical image channel. These aspects are crucial for capturing the nuanced
segmentation technology within the realm of clinical diagnostics. variations and complex structures often present in medical
However,medicalimagesegmentationfacesuniquechallenges,such images, which are essential for accurate diagnosis and analysis.
astheneedforprecisedelineationofcomplexanatomicalstructures, Swin-Unet (Cao et al., 2022) combines the Swin-transform block
variabilityacrosspatients,andthepresenceofnoiseandartifactsin with the U-net structure and achieves good results. Yet, adding
the images (Tran et al., 2023). These challenges necessitate the extensiveTransformerblocksinflatestheparametercountwithout
developmentofadvancedsegmentationtechniquesthatcancapture significantlyimprovingresults.Thisstudymerelystackedmultiple
fine-grained detailswhilemaintaining robustness and efficiency. Transformers to enhance models, resulting in inflated parameters
In the last decade, the traditional U-net structure has been andcomputationalcomplexitywithmarginalgainsinperformance.
widely employed in numerous segmentation tasks, yielding Moreover,somestudieshavespecificallyfocusedonincorporating
commendable outcomes. Notably, the U-Net model (Ronneberger position and channel attention mechanisms in medical image
etal.,2015),alongwithitsvariousenhancediterations,hasachieved segmentation. For instance, DA-DSUnet has been applied to
substantial success. ResUnet (Diakogiannis et al., 2020) emerged head-and-neck tumor segmentation, but it doesn’t combine
during this period, influenced by the residual concept. Similarly, Position Attention Module (PAM) and Channel Attention
UNet++ (Zhou et al., 2018) emphasizes enhancements in skip Module (CAM), nor does it discuss the potential filtering role of
connections. Moving beyond these CNN-based approaches, the DAblocksinskipconnections(Tangetal.,2021).Additionally,it
Transformer architecture introduces a completely new doesn’t leverage ViT for feature extraction. Another example is
perspective. The transformer (Vaswani et al., 2017), originally research on brain tumor segmentation, which, while applying DA
developed for sequence-to-sequence modeling in Natural blocks, limits its scope to brain tumors without validating other
Language Processing (NLP), has also found utility in the field of types of medical images (Sahayam et al., 2022). These studies
ComputerVision(CV).ViTssegmentimagesintopatchesandinput integrate DA blocks with other blocks but do not thoroughly
their embeddings into a transformer network for strong explore the role of DA in skip connections or optimize DA
performance. (Dosovitskiy et al., 2020). This signifies a trend of blocksfor theunique intricacies ofmedical imaging.
shiftingfromtraditionalCNNmodelstomoreflexibleTransformer However,Despitetheprogressmadebythesetransformer-based
models. While the above-mentioned U-Net structures have approaches, they often overlook the importance of integrating
enhanced the capabilities of models in segmentation tasks image-specific features, such as position and channel
(Ronneberger et al., 2015; Zhou et al., 2018; Diakogiannis et al., information, which are crucial for capturing the nuanced
2020), they do not integrate the more powerful feature extraction variations and complex structures in medical images. Moreover,
abilities inherent in the Transformer and attention mechanisms, the existing methods that incorporate dual attention mechanisms
which limits their potential for further improvement. On the one have not been optimized for the unique characteristics of medical
hand,severalstudieshavemadeprogressinimagesegmentationby imagery, leaving room for further improvement. To address these
leveragingDualAttention(DA)mechanismsforbothchannelsand limitations, we propose DA-TransUNet, which strategically
positions.TheDualAttentionNetwork(DANet)utilizesaPosition integrates the Dual Attention Block (DA-Block) into the
AttentionBlock(PAM)andChannelAttentionBlock(CAM)from transformer-based U-Net architecture, specifically tailored for
the DA Network for natural scene image segmentation (Fu et al., medical image segmentation.
2019). Thisresearchprimarily focuses onscene segmentation and In this research, our proposed model DA-TransUNet is an
doesnotexploretheuniquecharacteristicsofmedicalimagery.Also, innovative approach for medical image segmentation that
DAResUnet (Shi et al., 2020) introduces a dual attention block integrates the Transformer mechanism, specifically the Vision
combinedwitharesidualblock(Res-Block)inaU-netarchitecture Transformer (ViT) and a Dual Attention (DA) mechanism
for medical image segmentation, demonstrating significant within a U-Net architecture. First, the Transformer ViT is
improvements in this domain. However, in the realm of medical combined with DA in the encoder of the U-Net structure,
image segmentation, existing models, including those employing enhancing feature extraction capabilities by leveraging the
DualAttentionmechanisms,havenotyetextensivelyexploredthe detailed characteristics of medical images. This integration allows
optimalintegrationofDualAttentionwithTransformermodelsfor themodeltocapturebothlocalandglobalcontextualinformation,
enhanced feature extraction; this oversight representsasignificant whichisessentialforaccuratesegmentationofcomplexanatomical
FrontiersinBioengineeringandBiotechnology 02 frontiersin.org


## Page 3


Sunetal. 10.3389/fbioe.2024.1398237
structures. Then, to further refine feature extraction tailored to structure (Chen et al., 2021). Building on TransUNet, TransU-
medical images, DA is optimized for specific channels and Net++ incorporates attention mechanisms into both skip
incorporated into every module of the skip connections, enabling connections and feature extraction (Jamali et al., 2023). Swin-
themodeltoeffectivelyfilteroutirrelevantinformationandfocuson Unet (Cao et al., 2022) improves by replacing every convolution
the most discriminative features. The skip connections pass the block in U-net with Swin-Transformer (Liu et al., 2021). DS-
shallow positional information from the encoder, while the DA TransUNet proposes to incorporate the tif module (which is a
module refines the crucial detailed features. This targeted multi-scale module using Transformer) to the skip connection to
optimization is substantiated by extensive ablation studies, improvethemodel(Linetal.,2022).AA-transunetleveragesBlock
demonstrating its significance in improving the model’s Attention Model (CBAM) and Deep Separable Convolution to
performance. Lastly, this architecture has been rigorously tested further optimize TransUNet (Yang and Mehrkanoon, 2022).
across five medical image segmentation datasets and extensive TransFuse uses dual attention Bifusion blocks and AG to fuse
ablation studies, demonstrating its effectiveness and superiority features of two different parts of CNN and Transformer (Zhang
(Candemir et al., 2013; Jaeger et al., 2013; Bernal et al., 2015; etal.,2021).Numerousattentionmechanismshavebeenaddedto
Landman et al., 2015; Tschandl et al., 2018; Codella et al., 2019; U-netandTransUNetmodels,yetfurtherexplorationiswarranted.
Jha etal., 2020;Jha et al.,2021). Divergingfrompriorapproaches,ourexperimentintroducesadual
The main contributions of this article are summarized attentionmechanismandTransformermoduleintothetraditional
as follows: U-shaped encoder-decoder and skip connections, yielding
promisingresults.
1) The model of DA-TransUnet is proposed by integrating
Transformer ViT and Dual Attention in U-net
architecture’s encoder and skip connections. This design 2.2 Application of skip connections in
enhances feature extraction capabilities in better extracting medical image segmentation modeling
detailedfeatures ofmedical images.
2) WeproposeanoptimizedDualAttention(DA)Blockthatis Skip connections in U-net aim to bridge the semantic gap
designed for medical image segmentation with two key between the encoder and decoder, effectively recovering fine-
enhancements: the optimization of intermediate channel grained object details (Drozdzal et al., 2016; He et al., 2016;
configurations within the DA block, and its integration into Huang et al., 2017). There are three primary modifications to
each skip-connection layer for effectively filtering irrelevant skip connections: firstly, increasing their complexity (Azad et al.,
information. These are validated through comprehensive 2022a).U-Net++redesignedtheskipconnectiontoincludeaDense-
ablationexperiments. like structure in the skip connection (Zhou et al., 2018), and
3) The segmentation performance and generalization ability of U-Net3++(Huang et al., 2020) changed the skip connection to a
DA-TransUnet are validated on five medical datasets. In full-scale skip connection. Secondly, RA-UNet introduces a 3D
comparison to recent related studies, DA-TransUnet hybrid residual attention-aware method for precise feature
exhibits superior results in medical image segmentation, extraction in skipped connections (Jin et al., 2020). The third is a
demonstrating itseffectiveness in thisfield. combination ofencoder and decoder featuremaps: Analternative
extensiontotheclassicalskipconnectionwasintroducedinBCDU-
Therestofthisarticleisorganizedasfollows.Section2reviews Net with a bidirectional convolutional long-term-short-term
therelatedworksofautomaticmedicalimagesegmentation,andthe memory (LSTM) module was added to the skip connection
description of our proposed DA-TransUNet is given in Section 3. (Azad et al., 2019). Aligning with the second approach, we
Next,thecomprehensiveexperimentsandvisualizationanalysesare integrate Dual Attention Blocks into each skip connection layer,
conducted in Section 4. Finally, Section 5 makes a conclusion of enhancingdecoderfeatureextractionandtherebyimprovingimage
thewhole work. segmentation accuracy.
2 Related work 2.3 The use of attentional mechanisms in
medical images
2.1 U-net model
Attention mechanisms are essential for directing model focus
Recently,attentionmechanismshavegainedpopularityinU-net towardsrelevantfeatures,therebyenhancingperformance.Inrecent
architectures (Ronneberger et al., 2015). For example, Attention years, dual attention mechanisms have seen diverse applications
U-net incorporates attention mechanisms to enhance pancreas across multiple fields. In scene segmentation, the Dual Attention
localization and segmentation performance (Oktay et al., 2018); Network (DANet) employs position and channel attention
DAResUnet integrates both double attention and residual mechanisms to improve performance (Fu et al., 2019). A
mechanisms into U-net (Shi et al., 2020); Attention Res-UNet modularized DANs framework is presented that adeptly merges
explores the substitution of hard-attention with soft-attention visual and textual attention mechanisms (Nam et al., 2017). This
(Maji et al., 2022); Sa-unet incorporates a spatial attention cohesiveapproachenablesselectivefocusonpivotalfeaturesinboth
mechanism in U-net (Guo et al., 2021). Following this, types of data, thereby improving task-specific performance.
TransUNet innovatively combines Transformer and U-net Additionally, the introduction of the Dual Attention Module
FrontiersinBioengineeringandBiotechnology 03 frontiersin.org


## Page 4


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE1
IllustrationoftheproposeddualattentiontransformerU-Net(DA-TransUNet).Fortheinputmedicalimages,wefeedthemintoanencoderwith
transformerandDualAttentionBlock(DA-Block).Then,thefeaturesofeachofthethreedifferentscalesarepurifiedbyDA-Block.Finally,thepurifiedskip
connectionsarefusedwiththedecoder,whichsubsequentlyundergoesCNN-basedup-samplingtorestorethechanneltothesameresolutionasthe
inputimage.Inthisway,thefinalimagepredictionresultisobtained.
(DuATM) has been groundbreaking in the field of audio-visual Transformer layer and is further enriched by the DA-Block,
event localization. This model excels at learning context-aware which are exclusively introduced in this model architecture. In
feature sequences and performing attention sequence contrast, the decoder primarily employs conventional
comparisons in tandem, effectively incorporating auditory- convolutional mechanisms. For the optimization of skip
oriented visual attention mechanisms (Si et al., 2018). Moreover, connections, DA-Blocks serve as pivotal components within the
dual attention mechanisms have been applied to medical DA-TransUNet architecture. DA-Blocks filter irrelevant
segmentation, yielding promising results (Shi et al., 2020). The information in skip connections, enhancing image reconstruction
Multilevel Dual Attention U-net for Polyp Segment combines accuracy. In summary, in contrast to traditional convolutional
dual attention and U-net in medical image segmentation (Cai approaches and the extensive use of Transformers, DA-
et al.,2022). While significant progress has beenmade inmedical TransUNet uniquely leverages DA-Blocks for the extraction and
imagesegmentation,thereisstillampleroomforfurtherresearchto utilizationofimage-specificfeaturesofpositionandchannel.This
explorethepotentialofpositionandchannelattentionmechanism strategicincorporationsignificantlyelevatestheoverallperformance
in thefieldofmedical image segmentation. ofthemodel.
Compared to traditional U-Net architectures, DA-TransUNet
integrates the Transformer layer in the encoder to capture global
3 Methods
dependencies,whiletheU-Netreliessolelyonconvolutionallayers
forlocalfeatureextraction.Moreover,theinclusionofDA-Blocksin
In the subsequent section, we propose the DA-TransUNet the encoder and skip connections sets DA-TransUNet apart from
architecture, illustrated in Figure 1. We start with a both U-Net and Transformer-based models. These DA-Blocks
comprehensive overview ofthearchitecture. Next, we detailed the enable the extraction and utilization of image-specific position
architecture’s key components in the following order: the dual and channel features, enhancing the model’s ability to capture
attention blocks (DA-Block), the encoder, the skip connections, fine-grained detailscrucialfor medical image segmentation.
and thedecoder. ToelucidatetherationalebehindourproposedDA-TransUNet
model’s design, it’s imperative to consider the limitations and
strengths of both U-Net architectures and Transformers in the
3.1 Overview of DA-TransUNet context of feature extraction. While Transformers excel in global
featureextractionthroughtheirself-attentionmechanisms,theyare
InFigure1,thearchitectureofDA-TransUNetispresented.The inherently limited to unidirectional focus on positional attributes,
modelcomprisesthreecorecomponents:theencoder,thedecoder, thus neglecting multi-faceted feature perspectives. On the other
and the skip connections. In particular, the encoder fuses a hand,traditionalU-Netarchitecturesareproficientinlocalfeature
conventional convolutional neural network (CNN) with a extraction but lack the capability for comprehensive global
FrontiersinBioengineeringandBiotechnology 04 frontiersin.org


## Page 5


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE2
TheproposedDualAttentionBlock(DA-Block)isshownintheFigure.Thesameinputfeaturemapisinputintotwofeatureextractionlayers,oneis
thepositionfeatureextractionblockandtheotheristhechannelfeatureextractionblock,andfinally,thetwodifferentfeaturesarefusedtoobtainthe
finalDA-Blockoutput.
contextualization. To address these constraints, we integrate DA- weights are determined by the feature similarity between two
Blocks both preceding the Transformer layers and within the positions. Therefore, PAM is effective at extracting meaningful
encoder-decoder skip connections. This achieves two goals: spatial features.
firstly, it refines the feature map input to the Transformer, PAM initially takes a local feature, denoted as A ∈ RC×H×W (C
enabling more nuanced and precise global feature extraction; represents Channel, H represents, and W represents Width). We
secondly, the DA-Block in the skip connections optimize the thenfeedAintoaconvolutionallayer,resultinginthreenewfeature
transmitted features from the encoder, facilitating the decoder in maps,namely,B,C,andD,eachofsizeRC×H×W.Next,wereshapeB
reconstructing a more accurate feature map. Thus, our proposed andCtoRC×N,whereN=H×Wdenotesthenumberofpixels.We
architecture amalgamates the strengths and mitigates the performamatrixmultiplicationbetweenthetransposeofCandB
weaknesses of both foundational technologies, resulting in a and subsequently use a softmax layer to compute the spatial
robustsystem capable ofimage-specificfeature extraction. attention mapS ∈RN×N:
3.2 Dual attention block (DA-Block)
S
ji
(cid:1)
(cid:4)N i
e
(cid:1)1
xp
ex
(cid:1)
p
B
(cid:1)
i
B
·C
i ·
j
C
(cid:3)
j (cid:3)
(1)
Here,S measurestheimpactofthei-thpositiononthej-thposition.
As shown in the attached Figure 2, the Dual Attention Block ji
We then reshape matrix D to RC×N. A matrix multiplication is
(DA-Block) serves as a feature extraction module that integrates
image-specificfeaturesofpositionandchannel.Thisenablesfeature performedbetweenDandthetransposeofS,followedbyreshaping
the result to RC×H×W. Finally, we multiply it by a parameter α and
extractiontailoredtotheuniqueattributesoftheimage.Particularly
perform an element-wise sum operation with the features A to
in the context U-Net shaped architectures, the specialized feature
obtain thefinaloutputE ∈RC×H×W:
extraction capabilities of the DA-Block are crucial. While
Transformers are adept at using attention mechanisms to extract N
global features, they are notspecifically tailoredfor image-specific
E
j
(cid:1)α(cid:5)(cid:1)S
ji
D
i
(cid:3)+A
j
(2)
i(cid:1)1
attributes.Incontrast,theDA-Blockexcelsinbothposition-based
andchannel-basedfeatureextraction,enablingamoredetailedand Theweightαisinitializedas0andislearnedprogressively.PAM
accuratesetoffeaturestobeobtained.Therefore,weincorporateit has a strong capability to extract spatial features. It can be
into the encoder and skip connections to enhance the model’s inferred from Eq. 2 that the resulting feature E at each
segmentation performance. The DA-Block consists of two position is a weighted sum of the features across all positions
primary components: one featuring a Position Attention Module andoriginalfeatures,itpossessesglobalcontextualfeaturesand
(PAM), and the other incorporating a Channel Attention Module aggregates context based on the spatial attention map. This
(CAM),bothborrowedfromtheDualAttentionNetworkforscene ensures effective extraction of position features while
segmentation (Fuetal., 2019). maintaining global contextual information.
3.2.1 PAM (position attention module) 3.2.2 CAM (channel attention module)
As shown in Figure 3, PAM captures spatial dependencies AsshowninFigure4,thisisCAM,whichexcelsinextracting
between any two positions of feature maps, updating specific channel features. Unlike PAM, we directly reshape the original
features through a weighted sum of all position features. The feature A ∈ RC×H×W to RC×N, and then perform a matrix
FrontiersinBioengineeringandBiotechnology 05 frontiersin.org


## Page 6


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE4
ArchitectureofchannelattentionMechanism(CAM).
FIGURE3
ArchitectureofpositionattentionMechanism(PAM).
α(cid:6) 2(cid:1)Conv(cid:7)CAM(cid:7)α2(cid:8)(cid:8) (8)
m
as
u
o
lt
f
i
t
p
m
li
a
c
x
at
l
i
a
o
y
n
er
be
t
t
o
w
o
e
b
en
tai
A
n
a
t
n
h
d
e
i
c
t
h
s
a
t
n
ra
n
n
e
s
l
p
a
o
t
s
te
e
n
.S
ti
u
o
b
n
se
m
q
a
u
p
en
X
tly
∈
,w
R
e
C×
a
C
p
:
ply
After extracting
α(cid:6)
1 and
α(cid:6)
2 from the two layers of attention, the
output is obtained by aggregating and summing the two layers of
X
ji
(cid:1)
(cid:4)C i
e
(cid:1)1
xp
ex
(cid:1)
p
A
(cid:1)
i
A
·A
i ·
j
A
(cid:3)
j (cid:3)
(3) a
co
tt
n
en
vo
ti
l
o
u
n
tion
a
.
nd recovering the number of channels in one
output(cid:1)Conv(cid:1)α(cid:6) 1+α(cid:6) 2(cid:3) (9)
Here,x measurestheimpactofthei-thchannelonthej-thchannel.
ji
Next,weperformamatrixmultiplicationbetweenthetransposeofX
TooptimizetheDA-Blockformedicalimagesegmentation,we
andA,reshapingtheresulttoRC×H×W.Wethenmultiplytheresultby
fine-tunedthenumberofintermediatechannels.Thisoptimization
a scale parameter β and perform an element-wise sum operation
allowsthemodeltofocusonthemostcriticalfeatures,enhancingits
with Ato obtainthe finaloutputE ∈RC×H×W:
sensitivitytokeyinformationinthemedicalimages.Byadaptingthe
N DA-Blocktothespecificcharacteristicsofmedicalimages,weenable
E
j
(cid:1)β(cid:5)(cid:1)X
ji
A
i
(cid:3)+Aj (4)
the model to better capture the fine-grained details necessary for
i(cid:1)1
accuratesegmentation.Thistargetedoptimizationsetsourapproach
Likeα,βislearnedthroughtraining.SimilartoPAM,duringthe apartfrompreviousworks,whichoftenoverlooktheimportanceof
extraction of channel features in CAM, the final feature for each tailoringattentionmechanismstotheuniquedemandsofmedical
channelisgeneratedasaweightedsumofallchannelsandoriginal image segmentation.
features, thus endowing CAM with powerful channel feature
extraction capabilities.
3.3 Encoder with transformer and
3.2.3 DA (dual attention module) dual attention
AsshownintheFigure2,wepresentthearchitectureoftheDual
Attention Block (DA-Block). This architecture merges the robust As illustrated in Figure 1, the encoder architecture consists of
position feature extraction capabilities of the Position Attention four key components: convolution blocks, DA-Block, embedding
Module(PAM)withthechannelfeatureextractionstrengthsofthe layers, and transformer layers. Of particular significance is the
Channel Attention Module (CAM). Furthermore, when coupled inclusion of the DA block before the Transformer layer. This
with the nuances of traditional convolutional methodologies, the design is aimed at performing specialized image processing on
DA-Blockemergeswithsuperiorfeatureextractioncapabilities.DA- the post-convolution features, enhancing the Transformer’s
Block consists of two components, the first one is dominated by feature extraction for image content. While the Transformer
PAM and the second one is dominated by CAM. The first architecture plays a crucial role in preserving global context, the
component takes the input features and performs one DA block strengthens the Transformer’s capability to capture
convolution to scale the number of channels by one-sixteenth to image-specific features, enhancing its ability to capture global
get α1. This convolution operation not only simplifies feature contextual information in the image. This approach effectively
extraction by PAM but also helps to adjust the scale and combines global features with image-specific spatial and channel
dimension of features, making them more suitable for the characteristics.
subsequent attention mechanism computations. Following a Thefirstcomponentcomprisesthethreeconvolutionalblocksof
PAM feature extraction and another convolution,
α(cid:6)
1 is obtained, thearchitectureoftheU-Netanditsdiverse iterations,seamlessly
which further refinesthe extracted features. integratingconvolutionaloperationswithdownsamplingprocesses.
Eachconvolutionallayerhalvesthesizeoftheinputfeaturemapand
α1(cid:1)Conv(cid:7)input(cid:8) (5)
doubles its dimension, a configuration empirically found to
α(cid:6) 1(cid:1)Conv(cid:7)PAM(cid:7)α1(cid:8)(cid:8) (6)
maximize feature expressiveness while maintaining computational
efficiency.ThesecondcomponentusesDA-Blockextractfeaturesat
Theothercomponentisthesame,withtheonlydifferencebeingthat
bothpositionalandchannellevels,enhancingthedepthoffeature
thePAMblockisreplacedwithaCAMwiththefollowingformula:
representation while preserving the intrinsic characteristics of the
α2(cid:1)Conv(cid:7)input(cid:8) (7) inputmap.Thethirdcomponentistheembeddinglayerservesasa
FrontiersinBioengineeringandBiotechnology 06 frontiersin.org


## Page 7


Sunetal. 10.3389/fbioe.2024.1398237
criticalintermediary,enablingtherequisitedimensionaladaptation, feature map to its original dimensions. The third component: the
a prelude to the subsequent Transformer strata. The fourth threeupsamplingconvolutionblocksincrementallydoublethesize
component integrates Transformer layers for enhanced global of the input feature map in each step, effectively restoring the
feature extraction, beyond the reach of traditional CNNs. Putting image’sresolution.
the above parts together, it works as follows: the input image Puttingtheabovepartstogether,theworkflowbeginsbypassing
traverses three consecutive convolutional blocks, systematically the input image through convolution blocks and subsequently
expanding the receptive field to encompass vital features. performing upsampling to augment the size of the feature maps.
Subsequently, the DA-Block refines features through the These feature maps undergo a twofold size increase while their
application of both position-based and channel-based attention dimensionsarereducedbyhalf.Thefeaturesreceivedthroughthe
mechanisms. Following this, the remodeled features undergo a skipconnectionsarethenfused,followedbycontinuedupsampling
dimensionality transformation courtesy of the embedding andconvolution.Afterthreeiterationsofthisprocess,thegenerated
stratum before they are channeled into the Transformer feature map undergoes one final round of upsampling and is
framework for the extraction of all-encompassing global features. accurately restoredto itsoriginal sizebythe segmentation head.
This orchestrated progression safeguards the comprehensive Thanks to this architecture, the decoder demonstrates robust
retention of information across the continuum of successive decodingcapabilities,effectivelyrevitalizingtheoriginalfeaturemap
convolutional layers. Ultimately, the Transformer-generated using features fromboth theencoder and skip connections.
feature map is restructured and navigated through skip Furthermore, compared to other transformer-based approaches
connectionlayers to feed intothe decoder. thatextensivelyutilizetransformerblocksthroughoutthearchitecture,
By combining convolutional neural networks, transformer suchasSwin-Unet,DA-TransUNetachievesamorefavorablebalance
architectures, and dual-attention mechanisms, the encoder between performance and computational efficiency. The judicious
configuration culminates in a robust capability for feature integration of DA-Blocks in the encoder and skip connections
extraction, resulting in asymbioticpowerhouse ofcapabilities. allows DA-TransUNet to enhance feature representation while
maintainingamanageablecomputationalfootprint.
3.4 Skip-connections with dual attention
4 Experiments
SimilartootherU-structuredmodels,wehavealsoincorporated
skip connections between the encoder and decoder to bridge the Toevaluatetheproposedmethod,weperformedexperimentson
semantic gap that exists between them. To further minimize this Synapse (Landman et al., 2015), CVC-ClinicDB dataset (Bernal
semanticgap,weintroduceddual-attentionblocks(DA-Blocks),as et al., 2015), Chest X-ray mask and label dataset (Candemir
depictedinFigure1,ineachofthethreeskipconnectionlayers.This et al., 2013; Jaeger et al., 2013) Analysis, Kvasir SEG dataset (Jha
decision was based on our observation that traditional skip etal.,2020),Kvasir-Instrumentdataset(Jhaetal.,2021),2018ISIC-
connections often transmit redundant features, which DA-Blocks Task(Tschandletal.,2018;Codellaetal.,2019).Theexperimental
effectively filter. Integrating DA-Blocks into the skip connections results demonstrate that DA-TransUNet outperforms existing
allows them to refine the sparsely encoded features from both methods across all six datasets. In the following subsections, we
positional and channel perspectives, extracting more valuable firstintroducethedatasetandimplementationdetails.Thenshow
information while reducing redundancy. By doing so, DA-Blocks theresults oneach ofthesix datasets.
assist the decoder in more accurate feature map reconstruction.
Moreover, the inclusion of DA-Blocks not only enhances the
model’s robustness but also effectively mitigates sensitivity to 4.1 Datasets
overfitting, contributing to the overall performance and
generalization capabilityofthe model. 4.1.1 Synapse
The Synapse dataset consists of 30 scans of eight abdominal
organs. These eight organs include the left kidney, right kidney,
3.5 Decoder aorta,spleen,gallbladder,liver,stomachandpancreas.Therearea
totalof3779 axially enhanced abdominalclinical CT images.
As depicted in Figure 1, the right half of the diagram
corresponds to the decoder. The primary role of the decoder is 4.1.2 CVC—ClinicDB
toreconstructtheoriginalfeaturemapbyutilizingfeaturesacquired CVC-ClinicDB is a database of frames extracted from
from the encoder and those received through skip connections, colonoscopy videos, which is part of the Endoscopic Vision
employing operations like upsampling. Challenge. This is adataset ofendoscopic colonoscopy frames for
The decoder’s components include feature fusion, a the detection of polyps. CVC-ClinicDB contains 612 still images
segmentation head, and three upsampling convolution blocks. from29differentsequences.Eachimagehasitsassociatedmanually
The first component: feature fusion entails the integration of annotated groundtruth covering thepolyp.
feature maps transmitted through skip connections with the
existing feature maps, thereby assisting the decoder in faithfully 4.1.3 Chest Xray
reconstructingtheoriginalfeaturemap.Thesecondcomponent:the ChestXrayMasksandLabelsX-rayimagesandcorresponding
segmentation head is responsible for restoring the final output masks are provided. The X-rays were obtained from the
FrontiersinBioengineeringandBiotechnology 07 frontiersin.org


## Page 8


Sunetal. 10.3389/fbioe.2024.1398237
Montgomery County Department of Health and Human Services of-the-art algorithms. UCTansNet allocates skip connections
Tuberculosis Control Program, Montgomery County, Maryland, through the attention module in the traditional U-net model
UnitedStates.Thesetofimagescontains80anteriorandposterior (Wang et al., 2022a). TransNorm integrates the Transformer
X-rays, of which 58 X-rays are normal and 1702 X-rays are module into the encoder and skip connections of standard U-Net
abnormal with evidence of tuberculosis. All images have been de- (Azadetal.,2022b).AnovelTransformermodulewasdesignedand
identified and presented in DICOM format. The set contains a a model named MIM was built with it (Wang et al., 2022b). By
variety of abnormalities, including exudates and corneal extensively comparing our model with current state-of-the-art
morphology. It contains 138 posterior-anterior radiographs, of solutions, we intend to showcase its superior segmentation
which 80 radiographs were normal and 58 radiographs showed performance.
abnormal manifestations oftuberculosis.
4.2.2 Implementation details
4.1.4 Kvasir SEG WeimplementedDA-TransUNetusingthePyTorchframework
Kvasir SEG is an open-access dataset of gastrointestinal polyp and trained iton asingle NVIDIA RTX3090 GPU(Paszke etal.,
imagesandcorrespondingsegmentationmasks,manuallyannotated 2019). The model was trained with an image resolution of 256 ×
and verified by an experienced gastroenterologist. It contains 256 and a patch size of 16. We employed the Adam optimizer,
1000 polyp images and their corresponding groudtruth, the configured with a learning rate of 1e-3, momentum of 0.9, and
resolution of the images contained in Kvasir-SEG varies from weightdecayof1e-4.Allmodelsweretrainedfor500epochsunless
332×487to 1920 ×1072 pixels, and thefileformat is jpg. stated otherwise. In order to ensure the convergence of the
indicators, but due to different data set sizes, we used 50 epochs
4.1.5 Kvasir-instrument fortrainingonthetwodatasets,ChestXrayMasksandLabelsand
Kvasir-Instrument a gastrointestinal instrument Dataset. It ISIC2018-Task.
contains 590 endoscopic tool images and their groud truth mask, During the training phase on five datasets, including CVC-
theresolutionoftheimageinthedatasetvariesfrom720×576to ClinicDB,theproposedDA-TransUNetmodelistrainedinanend-
1280×1024,whichconsistsof590annotatedframescomprisingof to-endmanner.Itsobjectivefunctionconsistsofaweightedbinary
GIproceduretoolssuchassnares,balloons,biopsyforceps,etc.The cross-entropy loss function (BCE) and a Dice coefficient loss
fileformat is jpg. function. To facilitate training, the final loss function, termed
“Loss,”isformulated as follows:
4.1.6 2018ISIC-task
1 1
The dataset used in the 2018 ISIC Challenge addresses the Loss(cid:1) ×BCE+ ×DiceLoss (10)
2 2
challenges of skin diseases. It comprises a total of 2512 images,
withafileformatofJPG.Theimagesoflesionswereobtainedusing
ToensureafairevaluationoftheSynapsedataset,weutilizedthe
various dermatoscopic techniques from different anatomical sites pre-trainedmodel“R50-ViT”withinputresolutionandpatchsize
(excludingmucousmembranesandnails).Theseimagesaresourced setto224×224and16,respectively.Wetrainedthemodelusingthe
fromhistoricalsamplesofpatientsundergoingskincancerscreening SGDoptimizer,settingthelearningrateto0.01,momentumof0.9,
at multiple institutions. Each lesion image contains only a andweightdecayof1e-4.Thedefaultbatchsizewassetto24.The
primary lesion. lossfunctionemployedfortheSynapsedatasetisdefinedasfollows:
1 1
Loss(cid:1) ×Cross−EntropyLoss+ ×DiceLoss (11)
2 2
4.2 Implementation settings
This loss function balances the contributions of cross-entropy
4.2.1 Baselines andDicelosses,ensuringimpartialevaluationduringtestingonthe
In our endeavor to innovate in the field of medical image Synapse dataset.
segmentation, we benchmark our proposed model against an Whenusingthedatasets,weusea3to1ratio,where75%isthe
array of highly-regarded baselines, including the U-net, UNet++, trainingsetand25%isthetestset,toensureadequacyoftraining.
DA-Unet,AttentionU-net,andTransUNet.TheU-nethasbeena
foundational model in biomedical image segmentation 4.2.3 Model evaluation
(Ronneberger et al., 2015). Unet++ brings added sophistication In evaluating the performance of DA-TransUNet, we utilize a
with its implementation of intermediate layers (Zhou et al., comprehensive set of metrics including Intersection over Union
2018). The DA-Unet goes a step further by integrating dual (IoU),DiceCoefficient(DSC),andHausdorffDistance(HD).These
attention blocks, amplifying the richness of features extracted metrics are industry standards in computer vision and medical
(Cai et al., 2022). The Attention U-net employs an attention image segmentation, providing a multifaceted assessment of the
mechanism for improved feature map weighting (Oktay et al., model’saccuracy, precision, and robustness.
2018), and finally, the TransUNet deploys a transformer The choice of these metrics is based on their complementary
architecture, setting a new bar in segmentation precision (Chen nature and ability to capture different aspects of segmentation
et al., 2021). Through this comprehensive comparison with these quality. IoU and DSC measure the overlap between the predicted
eminent baselines, we aim to highlight the unique strengths and and ground truth segmentation masks, providing a global
expansive potential applications of our proposed model. assessment of the model’s ability to accurately identify and
Additionally, we benchmarked our model against advanced state- delineate target structures. HD, on the other hand, captures the
FrontiersinBioengineeringandBiotechnology 08 frontiersin.org


## Page 9


Sunetal. 10.3389/fbioe.2024.1398237
maximum distance between the predicted and ground truth InordertodemonstratethesuperiorityoftheDA-TransUNet
segmentation boundaries, ensuring that the predicted modelproposedinthispaper,weconductedthemainexperiments
segmentation closely adheres to the true boundaries of the target usingtheSynapsedatasetandcompareditwithits11state-of-the-
structures, even in the presence of small segmentation errors or artmodels (SOTA)(see Table1).
irregularities. AsshownintheFigure5,wecanseethattheaverageDSCand
IOU (Intersection over Union) is one of the commonly used average HD evaluation criteria are 79.80% and 23.48mm,
metricstoevaluatetheperformanceofcomputervisiontaskssuchas respectively, which are improved by 2.32% and 8.21mm,
objectdetection,imagesegmentationandinstancesegmentation.It respectively, compared with TransUNet, which indicates that
measuresthedegreeofoverlapbetweenthepredictedregionofthe our DA-TransUNet has better segmentation ability than
modelandtheactualtargetregion,whichhelpsustounderstandthe TransUNer in terms of overall segmentation results and organ
accuracyandprecisionofthemodel.Intargetdetectiontasks,IOUis edgeprediction.AsshownintheFigure6,ontheotherhand,we
usually used to determine the degree of overlap between the can see that DSC has the highest value of our model. Although
predicted bounding box (Bounding Box) and the real bounding HD is higher than Swin-Unet, it is still an improvement
box.Inimagesegmentationandinstancesegmentationtasks,IOUis compared to several newer models and TransUNet. The
usedtoevaluatethedegreeofoverlapbetweenthepredictedregion segmentation time for an image is 35.98ms for our DA-
and theground truthsegmentation region. TransUNet and 33.58ms for TransUNet, which indicates that
TP thereisnotmuchdifferenceinthesegmentationspeedbetween
IOU(cid:1)
FP+TP+FN
(12)
thetwomodels,butourDA-TransUNethasbettersegmentation
results.Inthesegmentationresultsof8organs,DA-TransUNet
The Dice coefficient (also known as the Sørensen-Dice outperforms TransUNet by 2.14%, 3.43%, 0.48%, 3.45%, and
coefficient, F1-score, DSC) is a measure of model performance in 4.11% for the five datasets of Gallbladder, right kidney, liver,
imagesegmentationtasks,andisparticularlyusefulfordealingwith spleen,andstomach,respectively.Thesegmentationrateforthe
classimbalanceproblems.Itmeasuresthedegreeofoverlapbetween pancreasisnotablyhigherat5.73%.Inacomparativeevaluation
thepredictedresultsandthegroundtruthsegmentationresults,and acrosssixdistinctorgans,DA-TransUNetdemonstratessuperior
is particularly effective whendealing with segmentation ofobjects segmentationcapabilitiesrelativetoTransUNet.Nevertheless,it
withunclearboundaries.TheDicecoefficientiscommonlyusedasa
exhibitsamarginaldecrementinthesegmentationaccuracyfor
measure of the model’s accuracy on the target region in image theaortaandleftkidneyby0.69%and0.17%,respectively.The
segmentation tasks, and is particularly suitable for dealing with modelachievesthebestsegmentationratesfortherightkidney,
relativelysmallor uneventarget regions. liver,pancreas,andstomach,indicatingsuperiorfeaturelearning
|P ∩T | 2|T∩P| capabilities on these organs.
Dice(P,T)(cid:1)
|P
1
|+|T
1
|
5Dice(cid:1)
|F|+|P|
(13)
To further confirm the better segmentation of our model
1 1
compared to TransUNet, we visualized the segmentation plots of
Hausdorff Distance(HD) isadistancemeasure formeasuring TransUNetandDA-TransUNet(seeFigure5).Fromtheyellowand
thesimilaritybetweentwosetsandiscommonlyusedtoevaluatethe purplepartsinthefirstcolumn,wecanseethatoursegmentation
performance of models in image segmentation tasks. It is effectisobviouslybetterthanthatofTransUNet;fromthesecond
particularly useful in the field of medical image segmentation to column,the extensionofpurple is betterthan thatofTransUNet,
quantify thedifference betweenpredicted andtruesegmentations. andthereisnovacancyinthebluepart;fromthethirdcolumn,there
The computation of Hausdorff distance captures the maximum isasemicircleintheyellowpart,andthevacancyinredissmaller
difference between the true segmentation result and the predicted than that of TransUNet, etc. It is evident that DA-TransUNet
segmentation result, and is particularly suitable for evaluating the outperforms TransUNet in segmentation quality. In summary,
performance ofsegmentation modelsin boundaryregions. DA-TransUNet significantly surpasses TransUNet in segmenting
H(A,B)(cid:1)max{maxa∈Aminb∈B (cid:3)a−b(cid:3),maxb∈Bmina∈A (cid:3)b−a(cid:3)} (14) theleftkidney,rightkidney,spleen,stomach,andpancreas.Italso
offers superior visualization performance in image segmentation.
WeevaluateusingbothDiceandHDintheSynapsedatasetand We simultaneously took DA-TransUNet infive datasets, CVC-
both Dice andIOU in otherdatasets. ClinicDB, Chest Xray Masks and Labels, ISIC2018-Task, kvasir-
instrument, and kvasir-seg, and compared it with some classical
models (see Table 2). In the table, the values of IOU and Dice of
4.3 Comparison to the state-of-the- DA-TransUNetarehigherthanTransUNetinallfivedatasets,CVC-
art methods ClinicDB, Chest Xray Masks and Labels, ISIC2018-Task, kvasir-
instrument, and kvasir-seg. Also DA-TransUNet has the best
4.3.1 Segmentation performance and comparison dataset segmentation in four of the five datasets. As seen in the
We have chosen U-net (Ronneberger et al., 2015), Res-Unet table, our DA-TransUNet has more excellent feature learning and
(Diakogiannis et al., 2020), TransUNet (Chen et al., 2021), imagesegmentationcapabilities.
U-Net++(Zhou et al., 2018), Att-Unet (Oktay et al., 2018), Wealsoshowtheresultsofimagesegmentationvisualizationof
TransNorm (Azad et al., 2022b), UCTransNet (Wang et al., 2022a), DA-TransUNetinthesefivedatasets,andwealsoshowtheresultsof
MultiResUNet (Ibtehaz and Rahman, 2020), swin-unet (Cao et al., thecomparisonmodelsforthecomparison.Thevisualizationresults
2022) and MIM (Wang et al., 2022b) to compare with our DA- for ChestX-ray Masksand Labels, Kvasir-Seg, Kvasir-Instrument,
TransUNet,andtheexperimentaldataaretabulatedbelow. ISIC2018-Task, and CVC-ClinicDB datasets are presented in
FrontiersinBioengineeringandBiotechnology 09 frontiersin.org


## Page 10


Sunetal. 10.3389/fbioe.2024.1398237
TABLE1ExperimentalresultsontheSynapsedataset.
Model Year DSC HD Aorta Gallbladder Kidney(L) Kidney(R) Liver Pancreas Spleen Stomach
↑ (%) ↓
U-net 2015 76.85 39.70 89.07 69.72 77.77 68.6 93.43 53.98 86.67 75.58
(Ronneberger
etal.,2015)
U-Net++(Zhou 2018 76.91 36.93 88.19 68.89 81.76 75.27 93.01 58.20 83.44 70.52
etal.,2018)
ResidualU-Net 2018 76.95 38.44 87.06 66.05 83.43 76.83 93.99 51.86 85.25 70.13
(Diakogiannis
etal.,2020)
Att-Unet(Oktay 2018 77.77 36.02 89.55 68.88 77.98 71.11 93.57 58.04 87.30 75.75
etal.,2018)
MultiResUNet 2020 77.42 36.84 87.73 65.67 82.08 70.43 93.49 60.09 85.23 75.66
(Ibtehazand
Rahman,2020)
TransUNet(Chen 2021 77.48 31.69 87.23 63.13 81.87 77.02 94.08 55.86 85.08 75.62
etal.,2021)
UCTransNet 2022 78.23 26.75 84.25 64.65 82.35 77.65 94.36 58.18 84.74 79.66
(Wangetal.,
2022a)
TransNorm(Azad 2022 78.40 30.25 86.23 65.1 82.18 78.63 94.22 55.34 89.50 76.01
etal.,2022b)
MIM(Wangetal., 2022 78.59 26.59 87.92 64.99 81.47 77.29 93.06 59.46 87.75 76.81
2022b)
swin-unet(Cao 2022 79.13 21.55 85.47 66.53 83.28 79.61 94.29 56.58 90.66 76.60
etal.,2022)
DA- 2023 79.80 23.48 86.54 65.27 81.70 80.45 94.57 61.62 88.53 79.73
TransUNet(Ours)
AverageRelative - 2.03 −9.00 −0.73% −1.09% 0.28% 5.21% 0.82% 4.86% 1.97% 4.5%
Improvement
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineachrespectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumberrepresentsthemethod
thatachievesthehighestscoreorlowesterroronthatparticularmetric,demonstratingitssuperiorperformancerelativetotheotherapproaches.
Figure7,Figure8,Figure9,Figure10,andFigure11,respectively.In is worth noting that the DA-Block itself is not computationally
the Figure, it can be seen that the segmentation effect of DA- intensive, as it only involves simple matrix multiplications and
TransUNet has a good performance. Firstly, DA-TransUNet has element-wise operations.
bettersegmentationresultsthanTransUNet.Inaddition,compared Table 3 compares the number of parameters, Dice Similarity
with the four classical models of U-net, Unet++, Attn-Unet, and Coefficient (DSC), and Hausdorff Distance (HD) between DA-
Res-Unet,DA-TransUNethasacertainimprovement.Itcanbeseen TransUNet and TransUNet.The incorporation of DA-Blocks leads
thattheeffectivenessofDA-TransUNetformodelsegmentationis toamodestincreaseof2.54%inthenumberofparameterscompared
not only confirmed in the Synapse dataset, but also in the five toTransUNet.Thisincrementalincreaseinparametersisjustifiable
datasets(CVC-ClinicDB,ChestXrayMasksandLabels,ISIC2018- considering the substantial performance gains achieved by DA-
Task,kvasir-instrument,kvasir-seg).WefurtherestablishthatDA- TransUNet, as demonstrated in our experimental results (Section
TransUNetexcelsinboth3Dand2Dmedicalimagesegmentation. 4). DA-TransUNet achieves an average improvement of 2.99% in
DSC and 25.9% in HD compared to TransUNet. The strategic
4.3.2 Computational complexity and efficiency placement of DA-Blocks allows for efficient feature refinement
The integration of DA-Blocks in the encoder and skip whilemaintainingareasonablemodelsize.
connections introduces additional computational overhead
compared to the standard TransUNet architecture. Let the
input feature map have a spatial resolution of H × W and C 4.4 Ablation study
channels. The computational complexity of the Position
Attention Module (PAM) is O(H2W2C), while the Channel We conducted ablation experiments on the DA-TransUNet
Attention Module (CAM) has a complexity of O(C2HW). As model using the Synapse dataset to discuss the effects of different
the DA-Block consists of both PAM and CAM, its overall factorsonmodelperformance.Specifically,itincludes:1)DA-Block
computational complexity is O(H2W2C+C2HW). However, it in Encoder.2) DA-Block in SkipConnection.
FrontiersinBioengineeringandBiotechnology 10 frontiersin.org


## Page 11


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE5
SegmentationresultsofTransUNetandDA-TransUNetontheSynapsedataset.
4.4.1 Effect of the DA-Block in encoder and skip baselinesawanincreasefrom77.48%to78.28%,HDindexdropped
connection from31.69mmto29.09mm.ThisindicatesthattheadditionofDA-
In this research (see Table 4), we conducted experiments to Blocksateachskipconnectionlayerprovidedthedecoderwithmore
assesstheimpactofintegratingDA-Blocksintotheencoderandskip refined features, mitigating feature loss during the upsampling
connections on the model’s segmentation performance. To be process, thereby reducing the risk of overfitting and enhancing
specific, we introduced DA-Blocks into each layer of the skip model stability. Furthermore, incorporating DA-Blocks into the
connections. The results demonstrated an improvement: the DSC encoder before the Transformer yielded an enhancement, with
FrontiersinBioengineeringandBiotechnology 11 frontiersin.org


## Page 12


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE6
LinechartofDSCandHDvaluesofseveraladvancedmodelsintheSynapsedataset.
TABLE2Experimentalresultsofdatasets(CVC-ClinicDB,ChestXrayMasksandLabels,ISIC2018-Task,kvasir-instrument,kvasir-seg).
CVC-ClinicDB Chest xray ISIC2018-task Kvasir- Kvasir-seg
masks and instrument
labels
Iou ↑ Dice↑ Iou↑ Dice↑ Iou↑ Dice ↑ Iou↑ Dice ↑ Iou↑ Dice ↑
U-net(Ronnebergeretal.,2015) 0.7821 0.8693 0.9303 0.9511 0.8114 0.8722 0.8957 0.9358 0.8012 0.8822
Attn-Unet(Oktayetal.,2018) 0.7935 0.8741 0.9274 0.9503 0.8151 0.876 0.8949 0.9359 0.7801 0.8661
Unet++(Zhouetal.,2018) 0.7847 0.8714 0.9289 0.9505 0.8133 0.873 0.8995 0.9389 0.7767 0.8657
ResUNet(Diakogiannisetal.,2020) 0.5902 0.7422 0.9262 0.9505 0.7651 0.8332 0.8572 0.9141 0.6604 0.7785
TransUNet(Chenetal.,2021) 0.8163 0.8901 0.9301 0.9535 0.8263 0.8878 0.8926 0.9363 0.8003 0.8791
DA-TransUNet(Ours) 0.8251 0.8947 0.9317 0.9538 0.8278 0.8888 0.8973 0.9381 0.8102 0.8847
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineachrespectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumberrepresentsthemethod
thatachievesthehighestscoreorlowesterroronthatparticularmetric,demonstratingitssuperiorperformancerelativetotheotherapproaches.
FIGURE7
ComparisonofqualitativeresultsbetweenDA-TransUNetandexistingmodelsonthetaskofsegmentingChestX-rayMasksandLabelsX-raydatasets.
theDSCbaselineincreasingfrom77.48%to78.87%,eventhoughthe inclusion of DA-Blocks both before the Transformer layer and
HDmetricdecreasedfrom31.69mmto27.71mm.Inconclusion, within the skip connections effectively boosts medical image
based onthefindingspresented inTable 4,we canassertthatthe segmentation capabilities.
FrontiersinBioengineeringandBiotechnology 12 frontiersin.org


## Page 13


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE8
ComparisonofqualitativeresultsbetweenDA-TransUNetandexistingmodelsonthetaskofsegmentingKvasir-Segdatasets.
FIGURE9
ComparisonofqualitativeresultsbetweenDA-TransUNetandexistingmodelsonthetaskofsegmentingKavsir-Instrumentdatasets.
FIGURE10
ComparisonofqualitativeresultsbetweenDA-TransUNetandexistingmodelsonthetaskofsegmenting2018ISIC-Taskdatasets.
4.4.2 Effect of adding DA-Blocks to skip metricdecreasedto25.80mmfrom27.71mm.AddingDA-Blocksto
connections in different layers thesecondandthirdlayersresultedinsomeprogress.WhenDA-Blocks
BuildingonthequantitativeresultsfromTable5,weexperimented wereintegratedacrossalllayers,therewasanimprovement,reflectedby
with various configurations of DA-Block placement across three a DSC of 79.80% and a HD of 23.48mm. In contrast to traditional
different layers of skip connections to identify the optimal architectures where skip connections indiscriminately pass features
architectural layout for enhancing the model’s performance. from the encoder to the decoder, our approach with DA-Blocks
Specifically, when DA-Blocks were added to just the first layer, the selectively improves feature quality at each layer. The results, as
DSCmetricimprovedto79.36%fromabaselineof78.87%,andtheHD corroborated by Table 5, reveal that introducing DA-Blocks to even
FrontiersinBioengineeringandBiotechnology 13 frontiersin.org


## Page 14


Sunetal. 10.3389/fbioe.2024.1398237
FIGURE11
ComparisonofqualitativeresultsbetweenDA-TransUNetandexistingmodelsonthetaskofsegmentingCVC-ClinicDBdatasets.
TABLE3ComparisonofmodelparametersandperformancebetweenDA-TransUNetandTransUNet.
Model Params Paramsincrease DSC improvement HDimprovement
TransUNet 105,276,066 - - -
DA-TransUNet 107,950,840 2.54% 2.99% 25.9%
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineachrespectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumberrepresentsthemethod
thatachievesthehighestscoreorlowesterroronthatparticularmetric,demonstratingitssuperiorperformancerelativetotheotherapproaches.
TABLE4EffectsofcombinatorialplacementofDA-Blocksintheencoderandthroughskipconnectionsonperformancemetrics.
Encoderwith DA Skipwith DA DSC ↑ HD↓
DA-TransUNet 77.48 31.69
DA-TransUNet √ 78.28 29.09
DA-TransUNet √ 78.87 27.71
DA-TransUNet √ √ 79.80 23.48
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineachrespectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumberrepresentsthemethod
thatachievesthehighestscoreorlowesterroronthatparticularmetric,demonstratingitssuperiorperformancerelativetotheotherapproaches.
TABLE5EffectsofincorporatingDA-Blockintheencoderandskipconnectionsatdifferentlayersonperformancemetrics.
1stlayer 2ndlayer 3rd layer DSC ↑ HD ↓
DA-TransUNet 78.87 27.71
DA-TransUNet √ 79.36 25.80
DA-TransUNet √ 78.65 23.43
DA-TransUNet √ 79.49 30.71
DA-TransUNet √ √ √ 79.80 23.48
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineachrespectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumberrepresentsthemethod
thatachievesthehighestscoreorlowesterroronthatparticularmetric,demonstratingitssuperiorperformancerelativetotheotherapproaches.
asinglelayerenhancesperformance,andthegreatestgainsareobserved 4.4.3 Effect of the number of intermediate
when applied across all layers. This indicates the effectiveness of channels in DA-Block
integrating DA-Blocks within skip connections for enhancing both BasedontheresultsshownintheTable6,weconductedadiscussion
featureextractionandmedicalimagesegmentation.Therefore,thetable regarding the size of the intermediate layer in the DA-Block, which
clearlysupportstheideathatlayer-wiseinclusionofDA-Blocksinskip demonstrates the effectiveness of convolutional layers from an
connections is an effective strategy for enhancing medical image experimentalperspective.TheoriginalDA-Blockhadanintermediate
segmentation. layersizethatisone-fourthoftheinputlayersize.However,sinceits
FrontiersinBioengineeringandBiotechnology 14 frontiersin.org


## Page 15


Sunetal. 10.3389/fbioe.2024.1398237
TABLE6EffectofthenumberofintermediatechannelsinDA-Block. TABLE7StatisticalanalysisofDSCimprovementsandmodelperformance.
1 2 4 8 16 32 DSC ↑ HD↓ Model Mean DSC ± SD 95% CI for DSC
DA-TransUNet √ 78.55 28.22 DA-TransUNet 79.80±5.01 [74.79,84.81]
DA-TransUNet √ 79.35 23.77 TransUNet 75.84±6.77 [69.06,82.61]
DA-TransUNet √ 79.71 25.90 Comparison of DSC improvements achieved by DA-
TransUNet and TransUNet relativetoU-net
DA-TransUNet √ 79.35 25.66
DA-TransUNet √ 79.80 23.48 Metric Mean 95%CI for t-Test
difference difference p-value
DA-TransUNet √ 79.71 24.45
Improvement 3.96 [0.40,7.53] 0.032
Theboldvaluesindicatethebestperformanceamongallthemethodscomparedineach
respectiveevaluationmetric.Specifically,foreachrowinatable,theboldnumber
representsthemethodthatachievesthehighestscoreorlowesterroronthatparticular
metric,demonstratingitssuperiorperformancerelativetotheotherapproaches. WefirstassessedthenormalityoftheDSCimprovementvaluesfor
both DA-TransUNet and TransUNet relative to U-Net using the
intendedapplicationisforroadscenesegmentationandnotspecifically Shapiro-Wilk test. The results showed p-values of 0.36 and 0.82 for
tailored for medical image segmentation, we deemed that setting the the improvements of DA-TransUNet and TransUNet, respectively.
intermediatelayersizetoone-fourthoftheinputlayersizemightnotbe Since both p-values are greater than 0.05, we cannot reject the null
suitable for the medical image segmentation domain. As seen in the hypothesis of normality. This indicates that the DSC improvement
graph,whenwesettheintermediatelayersizetobethesameastheinput valuesforbothDA-TransUNetandTransUNetrelativetoU-Netcan
size,theevaluationresultsshowaDSCof78.55%andHDof28.22mm. beconsideredapproximatelynormallydistributed.Wethenperformed
IntherelatedresearchDANet(Fuetal.,2019),wheretheintermediate a paired t-test to compare the significance of the improvements. As
layerwassettoone-fourthoftheinputlayer,theDSCresultwas79.71%, showninTable7,thetestyieldedat-statisticof2.45andap-valueof
andHDwas25.90mm.However,whenwefurtherreducedthesizeof 0.032, demonstrating a significant difference between the
the intermediate layer to one-sixteenth of the input layer size, we improvementsachievedbyDA-TransUNetandTransUNet.
observed an improvement in DSC to 79.80%, and HD decreased Moreover,tofurtherquantifythesuperiorityofDA-TransUNet
further to 23.48mm. It is evident that setting the intermediate layer overTransUNet,wecalculatedthe95%confidenceintervalforthe
toone-sixteenthoftheinputlayersizeismoresuitableformedicalimage difference in improvements between DA-TransUNet and
segmentationtasks.Thereductionintheintermediatelayersizecanhelp TransUNet. The results showed that the mean difference was
the model mitigate the risk of overfitting, optimize computational 3.96, with a standard deviation of 5.61, and the confidence
resources, and, given the precision requirements of medical image interval was [0.40, 7.53]. This means that, at a 95% confidence
segmentation tasks, enable the model to focus more on selecting the level,themagnitudeofthedifferenceinDSCimprovementsbetween
most crucial features, thereby enhancing sensitivity to critical DA-TransUNet and TransUNet lies between 0.40and 7.53.
informationforthetask. To provide a comprehensive overview of the models’
performance, we calculated the 95% confidence intervals for their
DSCscores.DA-TransUNetachievedameanDSCof79.80±5.01,
5 Discussion with a confidence interval of [74.79, 84.81], while TransUNet
achieved a mean DSC of 75.84 ± 6.77, with a confidence interval
Inthispresentstudy,wehavediscoveredpromisingoutcomes of[69.06,82.61].Theseresults,summarizedinTable7,suggestthat
fromtheintegrationofDA-BlockswiththeTransformerandtheir DA-TransUNetnotonlyachieveshigheraverageperformancebut
combination with skip-connections. Encouraging results were alsoexhibits moreconsistent results comparedto TransUNet.
consistently achievedacross allsix experimental datasets. The statistical analysis, confidence intervals, and the
quantification of the relative improvement provide strong
evidence for the superiority of DA-TransUNet over TransUNet
5.1Statisticalvalidationoftheimprovements inthetaskofmedicalimagesegmentation.Theseresultshighlight
by DA-TransUNet the effectiveness of our proposed approach and its potential to
advance the fieldofmedical image analysis.
Toenhancethecredibilityofourresultsandfurthervalidatethe
superiorityofDA-TransUNet,Weevaluatedtheperformanceofthe
modelsdiscussedintheExperimentSection4(U-Net,TransUNet, 5.2 Enhancing feature extraction and
and DA-TransUNet) on 12 subsets of the Synapse dataset, segmentation with DA-Blocks
constituting 40% of the total data, and obtained their Dice
Similarity Coefficients (DSC). It is important to note that both To start with, drawing from empirical results in Table 4, it is
DA-TransUNet and TransUNet are based on the U-Net demonstratedthattheintegrationofDA-Blockwithintheencoder
architecture, which serves as the baseline model. Therefore, using significantlyenhancesthefeatureextractioncapabilitiesaswellasits
U-NetasthebenchmarktoassesswhethertheimprovementsofDA- segmentation performance. In the landscape of computer vision,
TransUNet over TransUNet aresignificantis avalidapproach. Vision Transformer (ViT) has been lauded for its robust global
FrontiersinBioengineeringandBiotechnology 15 frontiersin.org


## Page 16


Sunetal. 10.3389/fbioe.2024.1398237
featureextractioncapabilities(Dosovitskiyetal.,2020).However,its improvements, particularly in the decoder section of the architecture.
falls short in specialized tasks like medical image segmentation, Thirdly,onepotentiallimitationofourDA-TransUNetarchitectureis
whereattentiontoimage-specificfeaturesiscrucial.Toremedythis, theriskoflosingfine-graineddetailsduringthetokenizationprocess,
in DA-TransUNet we strategically place DA-Blocks ahead of the which occurs after the convolution and pooling operations in the
Transformermodule.TheseDA-Blocksaretailoredtofirstextract encoder.This is particularlyconcerningformedicalimages with thin
and filter image-specific features, such as spatial positioning and andcomplexstructures,wherepreservingintricatedetailsiscrucialfor
channel attributes. Following this initial feature refinement, the accuratesegmentation.AlthoughourproposedintegrationoftheDual
processed data is then fed into the Transformer for enhanced Attention (DA) module before the Transformer in the encoder and
global feature extraction. This approach results in significantly withintheskipconnectionshelpsmitigatethisissuetosomeextent,as
improved feature learning and segmentation performance. In evidencedbytheimprovedsegmentationperformance,weacknowledge
summary, the strategic placement of DA-Blocks prior to the thattheremaystillberoomforfurtherenhancementincapturingand
Transformer layer constitutes a pioneering approach that retainingfine-grainedinformation.
significantly elevates both feature extraction efficacy and medical
image segmentation precision.
6 Conclusion
Morever,buildingonempiricaldatainTable5,ourintegrationof
DA-Blocks with skip connections significantly improves semantic
continuity and the decoder’s ability to reconstruct accurate feature Inthispaper,weinnovativelyproposedanovelapproachtoimage
maps.WhiletraditionalU-Netarchitectures(Ronnebergeretal.,2015) segmentation by integrating DA-Blocks with the Transformer in the
utilizeskipconnectionstobridgethesemanticgapbetweenencoderand architectureofTransUNet.TheDA-Blocks,focusingonimage-specific
decoder,ournovelincorporationofDualAttentionBlockswithinthe position and channel features, were further integrated into the skip
skip-connectionlayersyieldspromisingresults.ByincorporatingDA- connections to enhance the model’s performance. Our experimental
Blocksacrossskip-connectionlayers,wefocusonrelevantfeaturesand results, validated by an extensive ablation study, showed significant
filter out extraneous information, making the image reconstruction improvements in the model’s performance across various datasets,
processmoreefficientandaccurate.Insummary,thestrategicinclusion particularlytheSynapsedataset.
of DA-Blocks in skip connections represents a groundbreaking Our research revealed the potential of image-special features
approach that not only enhances feature extraction but also position and channel (DA-Block) in enhancing the feature
improvesthemodel’sperformanceinmedicalimagesegmentation. extraction capability and global information retention of the
Lastly, our extensive evaluation across six diverse medical image Transformer. The integration of DA-Block and Transformer
segmentation datasets demonstrates the effectiveness and substantially improved the model’s performance without creating
generalizability of the DA-TransUNet. The consistent improvements redundancy.Furthermore,theintroductionofDA-Blocksintoskip
over state-of-the-art methods (Table 1) highlight the impact of our connectionsnotonlyeffectivelybridges thesemanticgapbetween
targeted integration of the DA-Block. Moreover, the ablation studies theencoderanddecoder,butalsorefinesthefeaturemaps,leadingto
(4.4)providevaluableinsightsintotheindividualcontributionsofthe an enhanced image segmentation performance.
DA-Blockindifferentcomponentsofthearchitecture.Thesefindings Ourmodelalsohassomelimitations.Firstly,theintroductionof
notonlyunderscorethenoveltyofourapproachbutalsoshedlighton DAblocksincreasescomputationalcomplexity.Thisaddedcostmay
the importance of strategically integrating attention mechanisms for pose obstacles for real-time or resource-constrained applications.
enhancedmedicalimagesegmentation.TheDA-TransUNetrepresents Secondly, the decoder part of our model retains the original U-Net
a significant step forward in leveraging the power of attention architecture. Lastly, the utilization of image feature positions and
mechanisms and transformers for accurate and robust segmentation channelsisonlysuperficial,withdeeperexplorationpossible.
acrossawiderangeofmedicalimagingmodalities.Ourworkpavesthe Thisstudyhaspavedthewayforthefurtheruseofimage-special
wayforfurtherexplorationoftargetedattentionmechanismsinmedical featurespositionandchannel(DA-Block)inthefieldofmedicalimage
imageanalysisandhasthepotentialtoimpactclinicaldecision-making segmentation.Atthesametime,itprovidestheideaofleveragingimage
andpatientcare. characteristicstoachievehigh-precisionmedicalimagesegmentation.
Future work may focus on optimizing the decoder part of our
architecture and exploring methods to reduce the computational
5.3 Limitations and future directions complexity introduced by DA blocks without compromising the
model’s performance. We believe our approach can inspire future
Despitetheadvantages,ourmodelalsohassomelimitations.Firstly, researchinthedomainofmedicalimagesegmentationandbeyond.
the introduction of the DA-Blocks contributes to an increase in
computational complexity. This added cost could potentially be a
Data availability statement
hindranceinreal-timeorresource-constrainedapplications.Although
this increase in parameters is relatively modest considering the
performance gains achieved, it could still be a concern in resource- Publiclyavailabledatasetswereanalyzedinthisstudy.Thisdatacan
constrained scenarios or when dealing with very large-scale datasets. befoundhere:B.Landman,Z.Xu,J.E.Igelsias,M.Styner,T.Langerak,
Secondly, the decoder part of our model retains the original U-Net and A. Klein, ‘‘Segmentation outside the cranial vault challenge,’’ in
architecture.Whilethisdesignchoicepreservessomeoftheadvantages MICCAI: Multi Atlas Labeling Beyond Cranial Vault-Workshop
of U-Net, it also means that the decoder has not been specifically Challenge, 2015. J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D.
optimizedforourapplication.Thisleavesroomforfurtherresearchand Gil,C.Rodríguez,andF.Vilariño,‘‘Wm-dovamapsforaccuratepolyp
FrontiersinBioengineeringandBiotechnology 16 frontiersin.org


## Page 17


Sunetal. 10.3389/fbioe.2024.1398237
highlighting in colonoscopy: Validation vs. saliency maps from andediting.WK:Investigation,Validation,Writing–reviewand
physicians,’’ Computerized medical imaging and graphics, vol. 43, editing.ZX:Conceptualization,Writing–reviewandediting.JM:
pp. 99–111, 2015. N. Codella, V. Rotemberg, P. Tschandl, M. E. Conceptualization, Writing–review and editing. TR:
Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Methodology, Writing–review and editing. L-MN:
Marchettietal.,‘‘Skinlesionanalysistowardmelanomadetection2018: Conceptualization, Supervision, Writing–review and editing.
A challenge hosted by the international skin imaging collaboration JX: Investigation, Project administration, Resources,
(isic),’’ arXiv preprint arXiv:1902.03368, 2019. P. Tschandl, C. Supervision, Writing–review and editing.
Rosendahl, and H. Kittler, ‘‘The ham10000 dataset, a large collection
of multi-source dermatoscopic images of common pigmented skin
lesions,’’ Scientific data, vol. 5, no. 1, pp. 1–9, 2018. D. Jha, P. H. Funding
Smedsrud,M.A.Riegler,P.Halvorsen,T.deLange,D.Johansen,andH.
D.Johansen,‘‘Kvasir-seg:Asegmentedpolypdataset,’’inMultiMedia Theauthor(s)declarethatfinancialsupportwasreceivedfor
Modeling:26thInternationalConference,MMM2020,Daejeon,South theresearch,authorship,and/orpublicationofthisarticle.This
Korea, January 5–8, 2020, Proceedings, Part II 26. Springer, 2020, work was supported by The Soft Science Research Planning
pp. 451–462. D. Jha, S. Ali, K. Emanuelsen, S. A. Hicks, V. Project of Zhejiang Province under Grant 2024C35064 for the
Thambawita, E. GarciaCeja, M. A. Riegler, T. de Lange, P. T. project “Study on Performance Evaluation and Optimization
Schmidt, H. D. Johansen et al., ‘‘Kvasir-instrument: Diagnostic and Path of Digital Aging Transformation Driven by User
therapeutictoolsegmentationdatasetingastrointestinalendoscopy,’’in Experience.”
MultiMedia Modeling: 27th International Conference, MMM 2021,
Prague, Czech Republic, June 22–24, 2021, Proceedings, Part II 27.
Springer,2021,pp.218–229.S.Jaeger,A.Karargyris,S.Candemir,L. Conflict of interest
Folio,J.Siegelman,F.Callaghan,Z.Xue,K.Palaniappan,R.K.Singh,S.
Antani et al., ‘‘Automatic tuberculosis screening using chest The authors declare that the research was conducted in the
radiographs,’’ IEEE transactions on medical imaging, vol. 33, no. 2, absenceofanycommercialorfinancialrelationshipsthatcouldbe
pp.233–245,2013S.Candemir,S.Jaeger,K.Palaniappan,J.P.Musco,R. construed as apotential conflict ofinterest.
K. Singh, Z. Xue, A. Karargyris, S. Antani, G. Thoma, and C.
J. McDonald, ‘‘Lung segmentation in chest radiographs using
anatomical atlases with nonrigid registration,’’ IEEE transactions on Publisher’s note
medicalimaging,vol.33,no.2,pp.577–590,2013.
Allclaimsexpressedinthisarticlearesolelythoseoftheauthors
and do not necessarily represent those of their affiliated
Author contributions
organizations, or those of the publisher, the editors and the
reviewers. Any product that may be evaluated in this article, or
GS:Writing–originaldraft,Writing–reviewandediting.YP: claimthatmaybemadebyitsmanufacturer,isnotguaranteedor
Software, Visualization, Writing–original draft, Writing–review endorsed bythe publisher.
References
Azad,R.,Aghdam,E.K.,Rauland,A.,Jia,Y.,Avval,A.H.,Bozorgpour,A.,etal. Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., et al. (2021) Transunet:
(2022a)Medicalimagesegmentationreview:thesuccessofu-net.arXivpreprintarXiv: transformers make strong encoders for medical image segmentation. arXiv preprint
2211.14830. arXiv:2102.04306.
Azad,R.,Al-Antary,M.T.,Heidari,M.,andMerhof,D.(2022b).Transnorm: Codella,N.,Rotemberg,V.,Tschandl,P.,Celebi,M.E.,Dusza,S.,Gutman,D.,
transformer provides a strong spatial normalization mechanism for a deep et al. (2019) Skin lesion analysis toward melanoma detection 2018: a challenge
segmentation model. IEEE Access 10, 108205–108215. doi:10.1109/access.2022. hostedbytheinternationalskinimagingcollaboration(isic).arXivpreprintarXiv:
3211501 1902.03368.
Azad,R.,Asadi-Aghbolaghi,M.,Fathy,M.,andEscalera,S.(2019).“Bi-directional Diakogiannis,F.I.,Waldner,F.,Caccetta,P.,andWu,C.(2020).Resunet-a:adeep
convlstmu-netwithdensleyconnectedconvolutions,”inProceedingsoftheIEEE/CVF learning framework for semantic segmentation of remotely sensed data. ISPRS
internationalconferenceoncomputervisionworkshops. J.PhotogrammetryRemoteSens.162,94–114.doi:10.1016/j.isprsjprs.2020.01.013
Bernal,J.,Sánchez,F.J.,Fernández-Esparrach,G.,Gil,D.,Rodríguez,C.,andVilariño, Dosovitskiy,A.,Beyer,L.,Kolesnikov,A.,Weissenborn,D.,Zhai,X.,Unterthiner,T.,
F.(2015).Wm-dovamapsforaccuratepolyphighlightingincolonoscopy:Validation etal.(2020)Animageisworth16x16words:transformersforimagerecognitionatscale.
vs.saliencymapsfromphysicians.Comput.Med.imagingGraph.43,99–111.doi:10. arXivpreprintarXiv:2010.11929.
1016/j.compmedimag.2015.02.007 Drozdzal,M.,Vorontsov,E.,Chartrand,G.,Kadoury,S.,andPal,C.(2016).“The
Cai,Y.,Li,H.,Xin,J.,andSun,G.(2022).“Mlda-unet:multileveldualattentionunet importanceofskipconnectionsinbiomedicalimagesegmentation,”inInternational
forpolypsegmentation,”in202216thICMEInternationalConferenceonComplex WorkshoponDeepLearninginMedicalImageAnalysis,InternationalWorkshopon
MedicalEngineering(CME)(IEEE),372–376. Large-Scale Annotation of Biomedical Data and Expert Label Synthesis (Springer),
179–187.
Candemir,S.,Jaeger,S.,Palaniappan,K.,Musco,J.P.,Singh,R.K.,Xue,Z.,etal.
(2013). Lung segmentation in chest radiographs using anatomical atlases with Fu,J.,Liu,J.,Tian,H.,Li,Y.,Bao,Y.,Fang,Z.,etal.(2019).“Dualattentionnetwork
nonrigid registration. IEEE Trans. Med. imaging 33, 577–590. doi:10.1109/tmi. forscenesegmentation,”inProceedingsoftheIEEE/CVFconferenceoncomputer
2013.2290491 visionandpatternrecognition,3146–3154.
Cao,H.,Wang,Y.,Chen,J.,Jiang,D.,Zhang,X.,Tian,Q.,etal.(2022).“Swin-unet: Guo,C.,Szemenyei,M.,Yi,Y.,Wang,W.,Chen,B.,andFan,C.(2021).“Sa-unet:
unet-likepuretransformerformedicalimagesegmentation,”inEuropeanconference spatial attention u-net for retinal vessel segmentation,” in 2020 25th international
oncomputervision(Springer),205–218. conferenceonpatternrecognition(ICPR)(IEEE),1236–1242.
FrontiersinBioengineeringandBiotechnology 17 frontiersin.org


## Page 18


Sunetal. 10.3389/fbioe.2024.1398237
He,K.,Zhang,X.,Ren,S.,andSun,J.(2016).“Deepresiduallearningforimage Ronneberger, O., Fischer, P., and Brox, T. (2015). “U-net: convolutional
recognition,”inProceedingsoftheIEEEconferenceoncomputervisionandpattern networks for biomedical image segmentation,” in Proceedings, Part III
recognition,770–778. 18 Medical Image Computing and Computer-Assisted Intervention–MICCAI
co H nn u e a c n t g ed ,G c ., on L v iu o , lu Z ti ., on V a a l n n D et e w r o M rk a s a ,” ten in ,L. P , r a o n c d eed W in e g in s be o r f ge t r h , e K. IE Q E . E (2 c 0 o 1 n 7 f ) e . r “ e D nc e e nse o l n y 2 (S 0 p 1 r 5 i : ng 1 e 8 r t ) h ,2 I 3 n 4 t – e 2 rn 41 a . tional Conference, Munich, Germany, October 5-9, 2015
computervisionandpatternrecognition,4700–4708. Sahayam, S., Nenavath, R., Jayaraman, U., and Prakash, S. (2022). Brain tumor
Huang,H.,Lin,L.,Tong,R.,Hu,H.,Zhang,Q.,Iwamoto,Y.,etal.(2020).“Unet3+:a segmentationusingahybridmultiresolutionu-netwithresidualdualattentionand
full-scaleconnectedunetformedicalimagesegmentation,”inICASSP2020-2020IEEE deepsupervisiononmrimages.Biomed.SignalProcess.Control78,103939.doi:10.
1016/j.bspc.2022.103939
internationalconferenceonacoustics,speechandsignalprocessing(ICASSP)(IEEE),
1055–1059. Shi,Z.,Miao,C.,Schoepf,U.J.,Savage,R.H.,Dargis,D.M.,Pan,C.,etal.(2020).A
clinically applicable deep-learning model for detecting intracranial aneurysm in
Ibtehaz, N., and Rahman, M. S. (2020). Multiresunet: rethinking the u-net
computed tomography angiography images. Nat. Commun. 11, 6090. doi:10.1038/
architecture for multimodal biomedical image segmentation. Neural Netw. 121,
74–87.doi:10.1016/j.neunet.2019.08.025 s41467-020-19527-w
Si,J.,Zhang,H.,Li,C.-G.,Kuen,J.,Kong,X.,Kot,A.C.,etal.(2018).“Dualattention
Jaeger,S.,Karargyris,A.,Candemir,S.,Folio,L.,Siegelman,J.,Callaghan,F.,etal. matchingnetworkforcontext-awarefeaturesequencebasedpersonre-identification,”
(2013).Automatictuberculosisscreeningusingchestradiographs.IEEETrans.Med.
imaging33,233–245.doi:10.1109/tmi.2013.2284099 i
5
n
36
P
3
r
–
o
5
c
3
e
7
ed
2.
ingsoftheIEEEconferenceoncomputervisionandpatternrecognition,
Jamali, A., Roy, S. K., Li, J., and Ghamisi, P. (2023). Transu-net++: rethinking
Tang,P.,Zu,C.,Hong,M.,Yan,R.,Peng,X.,Xiao,J.,etal.(2021).Da-dsunet:
attentiongatedtransu-netfordeforestationmapping.Int.J.Appl.EarthObservation
dual attention-based dense su-net for automatic head-and-neck tumor
Geoinformation120,103332.doi:10.1016/j.jag.2023.103332 segmentation in mri images. Neurocomputing 435, 103–113. doi:10.1016/j.
Jha,D.,Ali,S.,Emanuelsen,K.,Hicks,S.A.,Thambawita,V.,Garcia-Ceja,E.,etal. neucom.2020.12.085
(2021).“Kvasir-instrument:Diagnosticandtherapeutictoolsegmentationdatasetin
I g n a t s e tr r o n i a n t t io es n t a in l a C l o e n n f d e o re sc n o c p e, y, M ”i M n M Pro 2 c 0 e 2 e 1 d , in P g r s a , gu P e a , rt C I z I ec 2 h 7 R M e u p l u ti b M lic e , di J a un M e o 2 d 2 e – li 2 n 4 g , : 2 2 0 7 2 th 1 an T ti r g a e n n ,T cl . a - s O si . fi ,a c n at d io L n e, u N si . n Q g . f K ea . t ( u 2 r 0 e 2 s 4 e ) x . t S r a a - c t t t e c d a: f a r n om svm bi - o b lo as g e ic d a a l p s p eq ro u a e c n h ci f n o g rt a u n m d o n r a t t - u c r e a l l l
(Springer),218–229. languageprocessing.Comput.Biol.Med.174,108408.doi:10.1016/j.compbiomed.2024.
108408
Jha,D.,Smedsrud,P.H.,Riegler,M.A.,Halvorsen,P.,deLange,T.,Johansen,D.,etal.
(2020).“Kvasir-seg:asegmentedpolypdataset,”inProceedings,PartII26MultiMedia
ap
T
p
r
r
a
o
n
ac
,
h
T
es
.-O
fo
.
r
,
l
V
un
o
g
, T
ca
.
n
H
ce
.
r
,
d
a
e
n
c
d
isi
L
on
e,
-m
N
a
.
ki
Q
n
.
g
K
an
.
d
(2
th
02
er
3
a
)
p
.
e
O
ut
m
ic
i
s
cs
d
-
e
b
v
a
e
s
l
e
o
d
pm
d
e
e
n
ep
t.B
le
r
a
ie
rn
fin
in
g
g
s
Modeling:26thInternationalConference,MMM2020,Daejeon,SouthKorea,January
5–8,2020(Springer),451–462. Funct.Genomics,elad031.doi:10.1093/bfgp/elad031
Tschandl,P.,Rosendahl,C.,andKittler,H.(2018).Theham10000dataset,alarge
Jin,Q.,Meng,Z.,Sun,C.,Cui,H.,andSu,R.(2020).Ra-unet:ahybriddeepattention-
collectionofmulti-sourcedermatoscopicimagesofcommonpigmentedskinlesions.
awarenetworktoextractliverandtumorinctscans.Front.Bioeng.Biotechnol.8, Sci.data5,180161–180169.doi:10.1038/sdata.2018.161
605132.doi:10.3389/fbioe.2020.605132
Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,etal.
Landman,B.,Xu,Z.,Igelsias,J.E.,Styner,M.,Langerak,T.,andKlein,A.(2015).
“Segmentationoutsidethecranialvaultchallenge,”inMICCAI:multiAtlaslabeling (2017).Attentionisallyouneed.Adv.neuralInf.Process.Syst.30.doi:10.48550/arXiv.
1706.03762
beyondcranialvault-workshopchallenge.
Wang,H.,Cao,P.,Wang,J.,andZaiane,O.R.(2022a).Uctransnet:rethinkingthe
Le,N.Q.K.(2024).Hematomaexpansionprediction:stillnavigatingtheintersection
ofdeeplearningandradiomics.Eur.Radiol.,1–3.doi:10.1007/s00330-024-10586-x
A
sk
A
ip
AI
co
C
n
o
n
n
e
f
c
.
t
A
io
r
n
ti
s
f.
i
I
n
nt
u
e
-
ll
n
.
e
3
t
6,
fr
2
o
4
m
41
a
–2
c
4
h
4
a
9
n
.
n
d
e
o
l-
i:
w
10
is
.
e
16
p
0
e
9
r
/
s
a
p
a
e
a
c
i.
t
v
iv
3
e
6i
w
3.
i
2
t
0
h
14
tr
4
ansformer. Proc.
Lin,A.,Chen,B.,Xu,J.,Zhang,Z.,Lu,G.,andZhang,D.(2022).Ds-transunet:dual Wang,H.,Xie,S.,Lin,L.,Iwamoto,Y.,Han,X.-H.,Chen,Y.-W.,etal.(2022b).“Mixed
1 sw –1 in 5. tr d a o n i s :1 fo 0 r .1 m 1 e 0 r 9 u /t - i n m e . t 2 f 0 o 2 r 2 m .3 e 1 d 7 i 8 c 9 al 9 i 1 magesegmentation.IEEETrans.Instrum.Meas.71, transformer u-net for medical image segmentation,” in ICASSP 2022-2022 IEEE
International Conference on Acoustics, Speech and Signal Processing (ICASSP)
Liu,Z.,Lin,Y.,Cao,Y.,Hu,H.,Wei,Y.,Zhang,Z.,etal.(2021).“Swintransformer: (IEEE),2390–2394.
h C i V er F ar i c n h te ic r a n l a v ti i o s n io a n lc tr o a n n fe sf r o e r n m ce er on us c in o g m s p h u i t f e te r d vi w si i o n n d , o 1 w 0 s 0 ,” 12 i – n 1 P 00 ro 2 c 2 e . edingsoftheIEEE/ for Ya n n o g w , c Y a . s , t a in n g dM tas e k h s r , k ” a i n n oo 2 n 0 , 2 S 2 .( I 2 n 0 te 2 r 2 n ) a . t “ i A on a a -t l ra Jo n i s n u t ne C t o :a n t f t e e r n e t n io ce n o au n gm Ne e u n r te a d lN tra et n w su o n rk e s t
Maji,D.,Sigedar,P.,andSingh,M.(2022).Attentionres-unetwithguideddecoderfor (IJCNN)(IEEE),01–08.
semanticsegmentationofbraintumors.Biomed.SignalProcess.Control71,103077. Zhang,Y.,Liu,H.,andHu,Q.(2021).“Transfuse:fusingtransformersandcnnsfor
doi:10.1016/j.bspc.2021.103077 medicalimagesegmentation,”inProceedings,PartI24MedicalImageComputingand
Nam,H.,Ha,J.-W.,andKim,J.(2017).“Dualattentionnetworksformultimodal Computer Assisted Intervention–MICCAI 2021: 24th International Conference,
reasoningandmatching,”inProceedingsoftheIEEEconferenceoncomputervision Strasbourg,France,September27–October1,2021(Springer),14–24.
andpatternrecognition,299–307.
Zhou,Z.,RahmanSiddiquee,M.M.,Tajbakhsh,N.,andLiang,J.(2018).“Unet++:a
Oktay,O.,Schlemper,J.,Folgoc,L.L.,Lee,M.,Heinrich,M.,Misawa,K.,etal.(2018) nested u-netarchitecture formedical imagesegmentation,”inProceedings 4Deep
Attentionu-net:learningwheretolookforthepancreas.arXivpreprintarXiv:1804.03999. LearninginMedicalImageAnalysisandMultimodalLearningforClinicalDecision
Support:4thInternationalWorkshop,DLMIA2018,and8thInternationalWorkshop,
Paszke,A.,Gross,S.,Massa,F.,Lerer,A.,Bradbury,J.,Chanan,G.,etal.(2019).
ML-CDS2018,HeldinConjunctionwithMICCAI2018,Granada,Spain,September
Pytorch:animperativestyle,high-performancedeeplearninglibrary.Adv.neuralInf. 20,2018(Springer),3–11.
Process.Syst.32.doi:10.48550/arXiv.1912.01703
FrontiersinBioengineeringandBiotechnology 18 frontiersin.org