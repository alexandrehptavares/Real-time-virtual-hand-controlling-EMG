%% Rede Neural para encontrar o gesto realizado de acordo com o sinal EMG
% Sinal EMG é extraído a cada 200 ms, corresponde a 80 pontos

function saida_rede_neural = RNA_MLP_deteccao_gesto(emgData,w1,b1,w2,b2,w3,b3,curSample,qtd_pontos_amostra)

[row, col] = find(isnan(emgData));
if isempty(row)
    row = qtd_pontos_amostra+1;
end
qtd_pontos_P1 = 520;
qtd_pontos_P1_elet_6 = 65;
qtd_entradas = qtd_pontos_amostra*8;
saida_rede_neural = 0;
x = zeros(qtd_entradas+qtd_pontos_P1+qtd_pontos_P1_elet_6,1);
% Entrada da Rede Neural
aux = emgData(row(1)-qtd_pontos_amostra:row(1)-1,:);
x(1:qtd_entradas) = aux(:); 
% aux(81,:) = [];
% x = zeros(199,1);
% [correlacao, lag] = xcorr(aux(:,6),aux(:,8));
% correlacao = correlacao*2;
% x = correlacao;

%% FFT

% CALCULO FFT
matriz_amostra = aux';
L_amostra = length(matriz_amostra);  % tamanho da amostra do sinal (número de pontos) for eletrodo=1:8
% FFT
n = 2^nextpow2(L_amostra);
dim = 2;
Y_amostra = fft(matriz_amostra,n,dim);
P2 = abs(Y_amostra/L_amostra);
P1 = P2(:,1:n/2+1);

matriz_fft = (P1*10);
x(qtd_entradas+1:qtd_entradas+qtd_pontos_P1) = matriz_fft(:);

Y_amostra_eletrodo6 = fft(matriz_amostra(6,:));
P2 = abs(Y_amostra_eletrodo6/L_amostra);
P1 = P2(:,1:n/2+1);

matriz_fft = (P1*15);
x(qtd_entradas+qtd_pontos_P1+1:qtd_entradas+qtd_pontos_P1+qtd_pontos_P1_elet_6) = matriz_fft(:);

%% Parâmentros RNA

% Dimensão: 3 Camadas
neuronios_escondidos_1 = 20;        % 1ª Camada Escondida
neuronios_escondidos_2 = 20;        % 2ª Camada Escondida
neuronios_saida = 4;                % Camada de saída

% [qtd_entradas, n_padroes_entrada] = size(x);

u1 = zeros(1,neuronios_escondidos_1);     % somatória neurônios 1ª camada
y1 = zeros(1,neuronios_escondidos_1);     % sáida neurônios 1ª camada
u2 = zeros(1,neuronios_escondidos_2);     % somatória neurônios 2ª camada
y2 = zeros(1,neuronios_escondidos_2);     % sáida neurônios 2ª camada
u3 = zeros(1,neuronios_saida);            % somatória neurônios 3ª camada
y3 = zeros(1,neuronios_saida);            % sáida neurônios 3ª camada

%% Cálculo da saída da RNA

% Cálculo das saídas da 1ª camada escondida
for j=1:neuronios_escondidos_1
    u1(j) = x'*w1(:,j) + b1(j);
    y1(j) = 2/(1+exp(-u1(j)))-1; % Sigmóide bipolar
end
% Cálculo das saídas da 2ª camada escondida
for j=1:neuronios_escondidos_2
    u2(j) = y1*w2(:,j) + b2(j);
    y2(j) = 2/(1+exp(-u2(j)))-1;
end
% Cálculo das saídas dos neurônios de saída
for j=1:neuronios_saida
    u3(j) = y2*w3(:,j) + b3(j);
    y3(j) = 2/(1+exp(-u3(j)))-1;
end

teste=0;
% 
% if y3(1) > 0.7 && y3(2) < -0.7 && y3(3) < -0.7
%     saida_rede_neural = 1; % mão aberta
% elseif y3(1) < -0.7 && y3(2) > 0.7 && y3(3) < -0.7
%     saida_rede_neural = 3; % mão pinca
% elseif y3(1) < -0.7 && y3(2) < -0.7 && y3(3) > 0.7
%     saida_rede_neural = 4; % mão fechada
% end
    
if y3(1) > 0.7 && y3(2) < -0.7 && y3(3) < -0.7 && y3(4) < -0.7
    saida_rede_neural = 1; % mão aberta
elseif y3(1) < -0.7 && y3(2) > 0.7 && y3(3) < -0.7 && y3(4) < -0.7
    saida_rede_neural = 2; % mão segurando
elseif y3(1) < -0.7 && y3(2) < -0.7 && y3(3) > 0.7 && y3(4) < -0.7
    saida_rede_neural = 3; % mão pinça
elseif y3(1) < -0.7 && y3(2) < -0.7 && y3(3) < -0.7 && y3(4) > 0.7
    saida_rede_neural = 4; % mão fechada
else
    saida_rede_neural = 333; % nenhum estado reconhecido
end

%% Envio da resposta da Rede Neural para o Unity

% tcpipClient = tcpip('127.0.0.1',55001,'NetworkRole','Client');
% set(tcpipClient,'Timeout',30);
% fopen(tcpipClient);
% a=num2str(saida_rede_neural);
% fwrite(tcpipClient,a);
% fclose(tcpipClient);

