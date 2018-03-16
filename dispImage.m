%------------------------------------------------------------
%dispImage v1.1  
%Heru Praptono,
%this tool is for displaying the images, from the dataset 
%data.mat                                          
%release (v1.0): 2018-03-13
%update:
%   v1.1: 2018-03-14
%     change note: add more 2 graphics on figure: histogram by bin, and notes on data
%------------------------------------------------------------
function dispImage(ds,trts,i)
%-----------------------------------------------------------
%Input(s):
%      ds (.mat)
%        name of dataset for the experiment
%      trts (1 or 2)
%        1 show train data, 2 shows test data
%      i
%        contains integer between 1 to the number of the data
%Output(s):
%      image's display
%      image's histogram
%      
%Example:
%      dispImage('data.mat',1,2)
%-----------------------------------------------------------

%-----------------------------------------------------------
%if you use MATLAB, remark below
%pkg load image
%-----------------------------------------------------------

data = load(ds);

X = data.dataset.train.images;
Y = data.dataset.train.labels;
X_t = data.dataset.test.images;
Y_t = data.dataset.test.labels;

if trts == 1
  ll = 'TRAIN DATA';
elseif trts == 2
  ll = 'TEST DATA' ;
  X = X_t;
  Y = Y_t;
end;

subplot(2,2,1)
imshow(reshape(X(i,:),28,28),[]);
title(strcat({ll},{' '},{'image:'},{' '},num2str(i),{' - '},{'class:'},{' '},{num2str(Y(i,1))}));


subplot(2,2,2)
%[counts, binloc]  = imhist(reshape(X(i,:),28,28));
%bar(binloc,counts,color = 'r')
imhist(reshape(X(i,:),28,28));
title('Image Histogram (per pixel map)');
xlabel('Gray Level (from 0 to 255)');
ylabel('Pixel count');


subplot(2,2,3)
hist(X(i,:),[0:50:255]);
title('Image Histogram (per pixel bin)');
xlabel('Gray level (bin size=50)');
ylabel('Pixel count');

subplot(2,2,4)
n_train = size(X,1);
n_test = size(X_t,1);
y = [n_train n_test];
bar(y, 'b');
title(strcat({'NUMBER OVERALL DATA'},{' '},{ds}));
xlabel('1: training data, 2: testing data');
ylabel('num of instances');