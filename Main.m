clc
clear
close all

%% Set Parameters

addpath('Images');
addpath('Old Methods');

nImages = 15; % number of Images
MaskImage = 1; %Images that used as mask

ShowVariable = 1;
ShowVariable2 = 1;

%% Read Images

Images = cell(nImages,1);
for i=1:nImages
   Images{i} = imread([num2str(i) '.jpg']);
end

Im_mask = imread([num2str(MaskImage) '_mask.jpg']);
Im_mask = im2bw(Im_mask); %#ok

%% Create Data

[Skin_RGB_Data,nonSkin_RGB_Data] = CreateData();

%% Skin Color Segmentation

Samples = Skin_RGB_Data';

[C,m] = covmatrix(Samples);
var = diag(C);
st = sqrt(var);
T = mean(st);

Detect_Im_euclidean = cell(1,nImages);
Detect_Im_mahalanobis = cell(1,nImages);

for i=1:nImages
   Im = Images{i};
   Detect_Im_euclidean{i} = colorseg('euclidean', Im, T*1.5, m);
   Detect_Im_mahalanobis{i} = colorseg('mahalanobis', Im, T*0.7, m, C);
end

%% morphological processing and detected skin area

if(ShowVariable==true)
   % euclidean method
   se = strel('disk',4);
   h = figure;
   h.Name = 'euclidean method';
   for i=1:nImages
      imc = imclose(Detect_Im_euclidean{i},se);
      imco = imopen(imc,se);
      imco = repmat(imco,[1,1,3]);
      im = uint8(double(Images{i}).*imco);
      subplot(4,floor( (size(Detect_Im_euclidean,2)-1)/4 )+1,i,'Parent',h);
      figure(h); imshow(im); 
      if (ShowVariable2==true)
         figure;
         subplot(1,2,1);imshow(Images{i});
         title('Original Image');
         subplot(1,2,2);imshow(im);
         title('euclidean method');
      end       
   end     
   
   % mahalanobis method
   se = strel('disk',4);
   h = figure;
   h.Name = 'mahalanobis method';
   for i=1:nImages
      imc = imclose(Detect_Im_mahalanobis{i},se);
      imco = imopen(imc,se);
      imco = repmat(imco,[1,1,3]);
      im = uint8(double(Images{i}).*imco);
      subplot(4,floor( (size(Detect_Im_mahalanobis,2)-1)/4 )+1,i,'Parent',h);
      figure(h); imshow(im); 
      if (ShowVariable2==true)
         figure;
         subplot(1,2,1);imshow(Images{i});
         title('Original Image');
         subplot(1,2,2);imshow(im);
         title('mahalanobis method');
      end       
   end   
end

%% Calculate Accuracy

M = Im_mask;
NN_eu = Detect_Im_euclidean{MaskImage};
NN_ma = Detect_Im_mahalanobis{MaskImage};

totalPixels = numel(Im_mask);

CDR_eu = ( sum(sum( (M==NN_eu) )) / totalPixels )*100; % correct detection rate 
FAR_eu = ( sum(sum( (M~=NN_eu)&(NN_eu==1) )) / totalPixels )*100; % false acceptance rate
FRR_eu = ( sum(sum( (M~=NN_eu)&(NN_eu==0) )) / totalPixels )*100; % false rejection rate

CDR_ma = ( sum(sum( (M==NN_ma) )) / totalPixels )*100; % correct detection rate 
FAR_ma = ( sum(sum( (M~=NN_ma)&(NN_ma==1) )) / totalPixels )*100; % false acceptance rate
FRR_ma = ( sum(sum( (M~=NN_ma)&(NN_ma==0) )) / totalPixels )*100; % false rejection rate

% display
disp('*************************************************');
disp(['CDR in euclidean method = ' num2str(mean(CDR_eu))]);
disp(' ');
disp(['FRR in euclidean method = ' num2str(mean(FRR_eu))]);
disp(' ');
disp(['FAR in euclidean method = ' num2str(mean(FAR_eu))]);
disp('*************************************************');

disp('*************************************************');
disp(['CDR in mahalanobis method = ' num2str(mean(CDR_ma))]);
disp(' ');
disp(['FRR in mahalanobis method = ' num2str(mean(FRR_ma))]);
disp(' ');
disp(['FAR in mahalanobis method = ' num2str(mean(FAR_ma))]);
disp('*************************************************');

%% Remove Paths

rmpath('Images');
rmpath('Old Methods');

