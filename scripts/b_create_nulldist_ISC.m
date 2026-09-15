clear all; clc;

% =========
% Load data
% =========
load('/Users/jadyn/Github/couple_sync/data/fNIRS/originstory_allcouples_oxy.mat');
subjectIDs = str2double(subjectInfo(:))';

% Check dimensions
[nSamples, nChannels, nSubjects] = size(allData);
nCouples = nSubjects / 2;

coupleIDs = unique(floor(subjectIDs / 10));
% Note: missing couple 1023 for origin story due to device failure

% ========
% Run ISC
% ========
nIter = 1000;

allIterISC = NaN(nIter, nChannels);

for iter = 1:nIter

    if mod(iter, 100) == 0
        fprintf("Running iteration %i \n", iter);
    end

    % ISC for every channel in each pair
    allPairsISC = NaN(nCouples, nChannels);

    for i = 1:nCouples

        thisCouple = coupleIDs(i);

        % Randomly select 1 or 2 from this couple
        suffix = randi([1 2]);
        thisSubID = thisCouple * 10 + suffix;

        % Find that subject from allData
        thisSubIdx = find(thisSubID == subjectIDs);
        thisSubTimeseries = allData(:, :, thisSubIdx);

        % Select random ID to calculate correlation
        otherCouples = coupleIDs(coupleIDs ~= thisCouple);
        randCouple = otherCouples(randi(numel(otherCouples)));

        % Randomly select 1 or 2 from the random couple
        randSubID = randCouple * 10 + suffix;

        % Find that random subject from all Data
        randSubIdx = find(randSubID == subjectIDs);
        randSubTimeseries = allData(:, :, randSubIdx);

        % Calculate ISC for each channel
        thisPairISC = NaN(1, nChannels);

        for j = 1:nChannels

            s1ThisROI = thisSubTimeseries(:, j);
            s2ThisROI = randSubTimeseries(:, j);

            corr = corrcoef(s1ThisROI, s2ThisROI, "Rows", "complete");
            thisPairISC(1,j) = corr(1,2);

        end

        % Stack across random pairs : allPairsISC = nPairs x nChannels
        allPairsISC(i, :) = thisPairISC;
    end

    % Average ISC across pairs
    allPairsISC = atanh(allPairsISC); % Fisher's z-transform
    nullISCZ = median(allPairsISC, "omitmissing"); % Find the median for each channel
    nullISCR = tanh(nullISCZ); % Inverse transform

    % Stack across iterations
    allIterISC(iter,:) = nullISCR;

end

% ==================
% Data save settings
% ==================
save("nullISC_originstory.mat", "allIterISC");
