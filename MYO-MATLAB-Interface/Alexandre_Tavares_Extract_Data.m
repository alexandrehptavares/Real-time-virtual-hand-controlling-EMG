%% Pseudo-real-time EMG data extraction from Myo to MATLAB
% Extracts EMG data from a MYO to MATLAB via C++
% Requires MYO Connect to be running
% Runs a C++ executable in a command prompt

clear; clc; close all % House keep

%% Genmeral Setup
lineLength = 54;        
numChannels = 8;
sampMax = 4000;
windowLength = 40;
curPosition = 0;

emgData = NaN([sampMax numChannels]); % Matrix for EMG data
mavData = NaN([(sampMax - windowLength) numChannels]); % Matrix for MAV feature

fileNameEMG = 'emg.txt'; % File for data transmission; again stop judging ^^
cmdWindowName = 'EMG Gather';

FileEMG = fopen(fileNameEMG,'w'); % Reset file
fclose(FileEMG);

system(['start /realtime "' cmdWindowName '" getMyoEmg.exe & exit &']) % Start (non-blocking) C thread
figure(1) % Do after cmd call to bring to foreground
set(gcf,'currentchar',']') % Used for exit on button press

%% Pause for handshake with myo connect and for data collection to begin
pause(1);

%% Leitura dos Pesos Sin�pticos obtidos do treinamento da RNA
w1 = load('w1');
w1 = w1.w1;
b1 = load('b1');
b1 = b1.b1;
w2 = load('w2');
w2 = w2.w2;
b2 = load('b2');
b2 = b2.b2;
w3 = load('w3');
w3 = w3.w3;
b3 = load('b3');
b3 = b3.b3;
resposta_rede_neural = 8888;
qtd_pontos_amostra = 80;
posicao = 'Abrindo';

%% Pseudo-realtime extraction
% Get first timestamp from file (and check data gathering is working)
FileEMG = fopen(fileNameEMG,'r');
fseek(FileEMG,curPosition,-1);
fileDataRaw = fgetl(FileEMG);
if fileDataRaw == -1
    system(['taskkill /f /fi "WindowTitle eq  ' cmdWindowName '" /T & exit'])
    close(gcf)
    disp('Data acquisition not active')
    return;
end
fileDataStrArray = strsplit(fileDataRaw,',');
startTime = str2double(fileDataStrArray(end));
fclose(FileEMG);

curSample = 1;
lastSample = 1;

while get(gcf,'currentchar')==']' % While no button has been pressed
    FileEMG = fopen(fileNameEMG,'r');
    fseek(FileEMG,curPosition,-1);
    
    fileDataRaw = ' ';
    while ischar(fileDataRaw) % Extract new data from file
        fileDataRaw = fgetl(FileEMG);
        if numel(fileDataRaw) ~= lineLength % Break if last line incomplete (and seek back to start of that line
            fseek(FileEMG,-numel(fileDataRaw) ,0);
            break;
        end
        
        fileDataStrArray = strsplit(fileDataRaw,',');
        curTime = str2double(fileDataStrArray(end));
        emgData(curSample,:) = str2double(fileDataStrArray(1:numChannels));
        
        if curSample >= windowLength % MAV feature extraction
            mavData(curSample - windowLength + 1,:) = mean(abs(emgData(curSample - windowLength + 1:curSample,:)));
        end      
                
        if mod(curSample,qtd_pontos_amostra)==0
            saida_rede_neural = RNA_MLP_deteccao_gesto(emgData,w1,b1,w2,b2,w3,b3,curSample,qtd_pontos_amostra);
            resposta_rede_neural = saida_rede_neural;  
            if resposta_rede_neural == 1
                posicao = 'ABERTA';
            elseif resposta_rede_neural == 2
                posicao = 'SEGURANDO';
            elseif resposta_rede_neural == 3
                posicao = 'PIN�A';
            elseif resposta_rede_neural == 4
                posicao = 'FECHADA';
            end
        end
       
        curSample = curSample + 1;    
               
    end
    curPosition = ftell(FileEMG);
    fclose(FileEMG);
    
    if curSample - lastSample == 0 % Don't waste time drawing if no new data
        continue;
    else
        lastSample = curSample;
    end   
    
    
    % Plots
    subplot(2,1,1)
    plot(emgData)
    ylim([-128 127])
    xlim([1 sampMax])
    title(['Sample frequency: ' num2str(curSample/(curTime - startTime))])
    xlabel('Samples')
    ylabel('Amplitude')
    
    subplot(2,1,2)
    plot(mavData)
    ylim([0 127])
    xlim([1 sampMax])
    title([num2str(windowLength) ' Sample Window MAV'])
    xlabel(['Resposta Rede Neural, posi��o da m�o:  ' num2str(posicao)])
    ylabel('Amplitude')
    drawnow
                                                                                  
    if curSample > sampMax % Clear arrays when large
        curSample = 1;
        lastSample = 1;
        emgData = NaN([sampMax numChannels]);
        mavData = NaN([(sampMax - windowLength) numChannels]);
        startTime = curTime;
    end
end

%% CLean up - target specific window made for this script
system(['taskkill /f /fi "WindowTitle eq  ' cmdWindowName '" /T & exit'])
close(gcf)

