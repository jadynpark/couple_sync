clear all; clc;

% =========
% Load data
% =========
load('/Users/jadyn/Github/couple_sync/data/fNIRS/originstory_allcouples_oxy.mat');

% Check dimensions
[nSamples, nChannels, nSubjects] = size(allData); 
nCouples = nSubjects / 2;

% Note: missing couple 1023 for origin story due to device failure

% =======
% Run ISC
% =======
% ISC for every channel in each couple
allCouplesISC = [];

for i = 1:nCouples

    % Each member's Id
    s1Id = 2*i - 1;
    s2Id = 2*i;

   % Calculate ISC for each channel
   thisCoupleISC = [];

    for j = 1:nChannels

        s1ThisChannel = allData(:, j, s1Id); 
        s2ThisChannel = allData(:, j, s2Id); 

        corr = corrcoef(s1ThisChannel, s2ThisChannel, "Rows", "complete");
        thisCoupleISC = [thisCoupleISC, corr(1,2)];

    end

    % Stack across couples : allCouplesISC = nCouples x nChannels
    allCouplesISC = [allCouplesISC; thisCoupleISC];

end

% Average ISC across couples
allCouplesISCZ = atanh(allCouplesISC); % Fisher's z-transform
trueISCZ = median(allCouplesISCZ, "omitmissing"); % Find the median for each channel
trueISCR = tanh(trueISCZ); % Inverse transform

% ==================
% Data save settings
% ==================
save("trueISC_originstory.mat", "trueISCR");

    
