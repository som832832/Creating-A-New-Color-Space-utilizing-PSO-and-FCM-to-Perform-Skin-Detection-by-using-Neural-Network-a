clc;
clear;
close all;

%% set Parameters

nImages = 3; % number of Images
PlotRange = [1,3]; % Range for plot
ImagesPath = 'Images';
AUC_Calculate = 1;

%% Create Data

[InputSkin,InputnonSkin]=CreateData();

TrainInputs=[InputSkin,InputnonSkin]';
TrainInputs=TrainInputs/255;
TrainTargets=[ ones(length(InputSkin),1);zeros(length(InputnonSkin),1) ];

TrainData=[TrainInputs,TrainTargets];
TrainData=TrainData(randperm(length(TrainData)),:);

% Input Images
Images = cell(nImages,1);
TestInputs = cell(nImages,1);
for i=1:nImages
   Images{i} = imread([ImagesPath, '\', 'P', num2str(i), '.jpg']); 
   R=Images{i}(:,:,1);
   G=Images{i}(:,:,2);
   B=Images{i}(:,:,3);
   RGB=double([R(:),G(:),B(:)]);
   TestInputs{i}=RGB/255;
end

% Mask Images
Masks = cell(nImages,1);
for i=1:nImages
   Mask = imread([ImagesPath, '\', 'P', num2str(i), '_M', '.jpg']);
   
   if(size(Mask,3)==3)
      Mask = rgb2gray(Mask);
   end
   
   Masks{i} = imbinarize(Mask,0.5);
end

%% Design ANFIS

Exponent=2;
MaxIter=100;
Maximprovement=1e-2;
DisplayValue=0;
FCMOption=[Exponent ...
           MaxIter ...
           Maximprovement ...
           DisplayValue];
nRules=10;        
fis=genfis3(TrainInputs,TrainTargets,'sugeno',nRules,FCMOption);

MaxEpoch=1000;
ErrorGoal=0;
InitialStepSize=0.05;
StepSizeDecreaseRate=0.9;
StepSizeIncreaseRate=1.1;
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);
% fuzzy(fis);

%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);

figure;

PlotResults(TrainTargets,TrainOutputs,'Train Data');

%% Apply ANFIS to Images Data

se=strel('disk',4);
TestImage = cell(nImages,1);
Scores = [];
Resp = [];
tic;

for i=1:nImages
   Score=evalfis(TestInputs{i},fis);
   Score = max(min(Score,1),0);
   
   TestOutputs=(Score>0.5);
   TestImage{i}=reshape(TestOutputs,[size(Images{i},1),size(Images{i},2)]);
   
   if(AUC_Calculate==1)            
       Scores = [Scores;Score]; %#ok
       M = Masks{i};
       Resp = [Resp;M(:)]; %#ok
   end
end

%% Calculate ROC

[X,Y,T,AUC,OPTROCPT] = perfcurve(Resp,Scores,1);

for i=1:length(X)
    Point = X(i)+Y(i);
    if(Point>=1)
       EER_1 = Y(i); 
       break; 
    end
end

figure;
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by ANFISn');

%% Show Outputs

EvalTime = toc;
EvalTime = EvalTime/nImages;

for i=PlotRange
   figure;
   subplot(2,2,1),imshow(Masks{i});
   title('Original Mask');
   
   subplot(2,2,2),imshow(Images{i});
   title('Original Image');
   
   subplot(2,2,3),imshow(TestImage{i});
   title('ANFIS Mask');
   
   imc=imclose(TestImage{i},se);
   imco=imopen(imc,se);
   imco=repmat(imco,1,1,3);
   OutImage_RGB=uint8(double(Images{i}).*imco);
   subplot(2,2,4),imshow(OutImage_RGB);
   title('ANFIS Output');
   
   saveas(gcf,[ImagesPath, '\', 'P', num2str(i), '_Out', '.jpg']);
end

%% Calculate Evaluation

CDR = zeros(1,nImages);
FAR = zeros(1,nImages);
FRR = zeros(1,nImages);
R = zeros(1,nImages);
P = zeros(1,nImages);
F = zeros(1,nImages);
FPR = zeros(1,nImages);
FNR = zeros(1,nImages);
TNR = zeros(1,nImages);
TDE = zeros(1,nImages);
ACC = zeros(1,nImages);

for i=1:nImages
    A = Masks{i};
    B = TestImage{i};
    totalPixels = numel(A);

    Tr = sum(sum( (A==B) ));    
    TP = sum(sum( (A==B)&(B==1) ));
    TN = sum(sum( (A==B)&(B==0) ));    
    FP = sum(sum( (A~=B)&(B==1) ));
    FN = sum(sum( (A~=B)&(B==0) ));
    
    % Evaluation protocols
    CDR(i) = Tr/totalPixels; % correct detection rate 
    FAR(i) = FP/totalPixels; % false acceptance rate
    FRR(i) = FN/totalPixels; % false rejection rate
    
    R(i) = TP/(TP+FN);
    P(i) = TP/(TP+FP);
    F(i) = (2*P(i)*R(i))/(P(i)+R(i));    
    FPR(i) = FP/(TN+FP);
    FNR(i) = FN/(TP+FN);
    TNR(i) = TN/(TN+FP);
    TDE(i) = FPR(i)+FNR(i);
    ACC(i) = (TP+TN)/(TP+TN+FP+FN);
end

clc;

disp('*************************************************');
disp(['Mean of CDR = ' num2str(mean(CDR))]);
disp(['Mean of FRR = ' num2str(mean(FRR))]);
disp(['Mean of FAR = ' num2str(mean(FAR))]);
disp(['Mean of R = ' num2str(mean(R))]);
disp(['Mean of P = ' num2str(mean(P))]);
disp(['Mean of FPR = ' num2str(mean(FPR))]);
disp(['Mean of FNR = ' num2str(mean(FNR))]);
disp(['Mean of TNR = ' num2str(mean(TNR))]);
disp(['Mean of TDE = ' num2str(mean(TDE))]);
disp(['Mean of ACC = ' num2str(mean(ACC))]);
disp(['Mean of AUC = ' num2str(AUC)]);
disp(['Mean of 1-EER = ' num2str(EER_1)]);
disp(['Mean of Time (sec) = ' num2str(EvalTime)]);
disp('*************************************************');
