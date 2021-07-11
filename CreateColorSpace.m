clc
clear
close all
format shortG

%% Set parameters

Algorithm = 'PSO';
Mode = 'N'; % 'L' for Linear Color Space - 'N' for nonLinear Color Space

nImages = 15;
TrainImage = 1; 

LoadData = 1;

addpath('Images');
addpath(['Heuristic Algorithms\',Algorithm]);

%% Read Images

Im = imread([num2str(TrainImage),'.jpg']);
Im = double(Im)/256;

Mask_In = imread([num2str(TrainImage),'_Mask.jpg']);
Mask_In = im2bw(Mask_In); %#ok

%% optimize color space by Heuristic Algorithm

if(LoadData==0)
   % Create W matrix
   feval(Algorithm);

   % Save W matrix
   save(['Data/Wmatrix_',Mode,'_',Algorithm,'.mat'],'W');
else
   % Load W matrix
   load(['Data/Wmatrix_',Mode,'_',Algorithm,'.mat']);
end

%% Display

Im2 = cell(1,nImages+1);

for i=1:nImages
   Im2{i} = imread([num2str(i),'.jpg']);
   Im2{i} = double(Im2{i})/256;
end

for i=1:nImages+1
   % convert color space
   [r,c,~] = size(Im2{i});
   RGB = reshape(Im2{i},r*c,3);

   Y = rgb2newColorSpace(RGB,W,Mode);
   
   Imnew = reshape(Y,r,c,3);

   figure;
   subplot(1,2,1); imshow(Im2{i}); title('RGB color space');
   subplot(1,2,2),imshow(mat2gray(Imnew)); title('New color space');
end

%% Remove Paths

rmpath('Images');
rmpath(['Heuristic Algorithms\',Algorithm]);
