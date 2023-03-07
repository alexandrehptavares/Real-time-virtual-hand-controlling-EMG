clear; clc;
close all;
format long;
%% Treinamento de uma Rede Neural Artificial MLP de 3 camadas
% Entradas: Sinais EMG de uma MYO Armband
% Saídas: Estados da mão (aberta ou fechada)
%% Parâmetros REDE NEURAL MLP

% Dimensão: 3 Camadas
neuronios_escondidos_1 = 20;        % 1ª Camada Escondida
neuronios_escondidos_2 = 20;        % 2ª Camada Escondida
neuronios_saida = 4;                % Camada de saída

frequencia_amostragem = 200;    % Hz
duracao_sinal = 120;            % segundos
tamanho_vetor_entradas = frequencia_amostragem*duracao_sinal;
qtd_pontos_amostragem = frequencia_amostragem*0.4;  % referente a 400 ms

%% Lendo os sinais gravados em .txt

mao_aberta = load('treinamento_emg_MAO_ABERTA_2_min_200_Hz_2.txt');
mao_fechada = load('treinamento_emg_MAO_FECHADA_2_min_200_Hz.txt');
mao_segurando =  load('treinamento_emg_MAO_SEGURANDO_2_min_200_Hz(2).txt');
mao_pinca =  load('treinamento_emg_MAO_PINCA_2_min_200_Hz_2.txt');

%% Tratamento do sinal, Entradas e Targets

% Normalizar sinal entre 1 e -1
mao_aberta = mao_aberta/128;
mao_segurando = mao_segurando/128;
mao_pinca = mao_pinca/128;
mao_fechada = mao_fechada/128;
% Reduzir as matrizes para a duração do sinal imposta anteriormente
mao_aberta(tamanho_vetor_entradas+1:length(mao_aberta),:) = [];
mao_segurando(tamanho_vetor_entradas+1:length(mao_segurando),:) = [];
mao_pinca(tamanho_vetor_entradas+1:length(mao_pinca),:) = [];
mao_fechada(tamanho_vetor_entradas+1:length(mao_fechada),:) = [];
% Eliminar última coluna das 2 matrizes (referente ao Timestemp)
mao_aberta(:,9) = [];
mao_segurando(:,9) = [];
mao_pinca(:,9) = [];
mao_fechada(:,9) = [];

% ENTRADAS da Rede Neural
x_inicial = [mao_aberta; mao_fechada]; % [mao_aberta; mao_segurando; mao_pinca; mao_fechada];
n_padroes_entrada = length(x_inicial)/qtd_pontos_amostragem;
qtd_entradas = qtd_pontos_amostragem*8;      % 8 colunas: 8 eletrodos
qtd_pontos_correlacao = 10176;
% x = zeros(qtd_entradas*2+qtd_pontos_correlacao,n_padroes_entrada);
% x = zeros(qtd_pontos_correlacao,n_padroes_entrada);
qtd_pontos_P1 = 520;
qtd_pontos_P1_elet_6 = 65;
x = zeros(qtd_entradas+qtd_pontos_P1,n_padroes_entrada);
% x = zeros(qtd_entradas+qtd_pontos_P1+qtd_pontos_correlacao,n_padroes_entrada);
% Separar os sinais de entrada por amostras e adicionar suas FFTs
for i=0:n_padroes_entrada-1
    %% ENTRADAS DOMÍNIO DO TEMPO
    aux = x_inicial(qtd_pontos_amostragem*i + 1 : qtd_pontos_amostragem*i + qtd_pontos_amostragem,:);
    x(1:qtd_entradas,i+1) = aux(:);
    % CALCULO FFT
    matriz_amostra = aux';
    L_amostra = length(matriz_amostra);  % tamanho da amostra do sinal (número de pontos)
%     posicao_inicial_gravacao_em_x = qtd_entradas+1;
%     posicao_final_gravacao_em_x = posicao_inicial_gravacao_em_x + 99;
    %         for eletrodo=1:8
    % FFT
    Fs = frequencia_amostragem;
    n = 2^nextpow2(L_amostra);
    dim = 2;
    Y_amostra = fft(matriz_amostra,n,dim);
    P2 = abs(Y_amostra/L_amostra);
    P1 = P2(:,1:n/2+1);
    
    matriz_fft = (P1*15);
    x(qtd_entradas+1:qtd_entradas+qtd_pontos_P1,i+1) = matriz_fft(:);
    f = 0:(Fs/n):(Fs/2-Fs/n);
        
    %
    % %             P1(:,2:end-1) = 2*P1(:,2:end-1);
    %             %             Y_amostra = fft(matriz_amostra(:,eletrodo));
    %             %             Y_amostra = fftshift(Y_amostra);
    %             % Vetor de frequências
    % %             f_amostra = frequencia_amostragem*(1:L_amostra)/L_amostra;
    %             f_amostra = (-L_amostra/2:L_amostra/2-1)*frequencia_amostragem/L_amostra;
    %             % Potências
    %             abs_FFT = abs(Y_amostra);
    % %             P1_amostra = P2_amostra(1:L_amostra);
    % %             P1_amostra = zeros(8,L_amostra);
    % %             for eletrodoS=1:8
    % %                 P1_amostra(eletrodoS,:) = P2_amostra(1:L_amostra,eletrodoS);
    % %             end
    % %             matriz_fft = (P1_amostra*15)';
    % %             x(qtd_entradas+1:qtd_entradas*2,i+1) = matriz_fft(:);
    %             x(posicao_inicial_gravacao_em_x:posicao_final_gravacao_em_x,i+1) = abs_FFT;
    %             posicao_inicial_gravacao_em_x = posicao_inicial_gravacao_em_x + 100;
    %             posicao_final_gravacao_em_x = posicao_final_gravacao_em_x + 100;
    %         end
    
    %% CORRELAÇÂO dos eltrodos 6 e 8 (eletrodos com maior diferença de sinais)
    
%     correlacao = xcorr(aux);
%     %     correlacao = correlacao*2;
%     %     % correlacao = xcorr(aux);
%     correlacao = correlacao';
%     correlacao = correlacao*0.5;
%     %     correlacao = correlacao*0.5;
%     %     %     correlacao = correlacao/(max(correlacao));  % normalizar entre 0 e 1
%     x(qtd_entradas+qtd_pontos_P1+1:length(x),i+1) = correlacao(:);
end

% TARGETS da Rede Neural
t = zeros(n_padroes_entrada,2);
for i=1:n_padroes_entrada
        if i<=(n_padroes_entrada/2)
            t(i,:) = [1, -1]; % Mão ABERTA
        else
            t(i,:) = [-1, 1]; % Mão FECHADA
        end
end

%% ===================== Início Plotagens =================================
%% Plotagem dos sinais (domínio do tempo)

% Plotagem de todos os eletrodos juntos
figure();
plot(mao_aberta); ylim([-1 1]); title('Mão aberta');
figure();
plot(mao_fechada); ylim([-1 1]); title('Mão fechada');
%% Plotagem das FFTs

% L = length(mao_aberta);
% Fs = frequencia_amostragem;
% % n = 2^nextpow2(L);
% dim = 2;
% figure();
% Y_mao_aberta = fft(mao_aberta);
% P2 = abs(Y_mao_aberta/L);
% P1 = P2(1:L/2);
% plot(P1); title('FFT Mão Aberta');
% figure();
% Y_mao_fechada = fft(mao_fechada);
% P2 = abs(Y_mao_fechada/L);
% P1 = P2(1:L/2);
% plot(P1); title('FFT Mão Fechada');
%% Plotagem das convoluções de cada sinal em comparação à MÃO ABERTA
% figure();
% for i = 1:8
%     subplot(4,2,i);
%     convolucao = conv(mao_aberta(:,i),mao_aberta(:,i));
%     plot(convolucao); ylim([-1 1])
%     title(['Convolução Mão ABERTA, eletrodo: ' num2str(i)]);
% end
% figure();
% for i = 1:8
%     subplot(4,2,i);
%     convolucao = conv(mao_segurando(:,i),mao_aberta(:,i));
%     plot(convolucao); ylim([-1 1])
%     title(['Convolução Mão SEGURANDO, eletrodo: ' num2str(i)]);
% end
% figure();
% for i = 1:8
%     subplot(4,2,i);
%     convolucao = conv(mao_pinca(:,i),mao_aberta(:,i));
%     plot(convolucao); ylim([-1 1])
%     title(['Convolução Mão PINÇA, eletrodo: ' num2str(i)]);
% end
% figure();
% for i = 1:8
%     subplot(4,2,i);
%     convolucao = conv(mao_fechada(:,i),mao_aberta(:,i));
%     plot(convolucao); ylim([-1 1])
%     title(['Convolução Mão FECHADA, eletrodo: ' num2str(i)]);
% end
%% Plotagem dos eletrodos seperadamente
% figure();
% for i = 1:8
%     subplot(4,2,i);
%     plot(mao_aberta(:,i)); ylim([-1 1])
%     title(['Mão ABERTA, eletrodo: ' num2str(i)]);
% end
% figure();
% for i=1:8
%     subplot(4,2,i);
%     plot(mao_segurando(:,i)); ylim([-1 1])
%     title(['Mão SEGURANDO, eletrodo: ' num2str(i)]);
% end
% figure();
% for i=1:8
%     subplot(4,2,i);
%     plot(mao_pinca(:,i)); ylim([-1 1])
%     title(['Mão PINÇA, eletrodo: ' num2str(i)]);
% end
% figure();
% for i=1:8
%     subplot(4,2,i);
%     plot(mao_fechada(:,i)); ylim([-1 1])
%     title(['Mão FECHADA, eletrodo: ' num2str(i)]);
% end
%% Plotagem das Xcorr dos sinais
% figure();
% x_cor_1 = xcorr(mao_aberta);
% plot(x_cor_1); title('Xcorr Mão aberta');
% figure();
% x_cor_2 = xcorr(mao_segurando);
% plot(x_cor_2); title('Xcorr Mão segurando');
% figure();
% x_cor_3 = xcorr(mao_pinca);
% plot(x_cor_3); title('Xcorr Mão pinça');
% figure();
% x_cor_4 = xcorr(mao_fechada);
% plot(x_cor_4);  title('Xcorr Mão fechada');
%% Plotagem das amostragens do sinal

num_salto_amostras = n_padroes_entrada/8;
% num_salto_amostras = int8(num_salto_amostras);
figure();
for i=1:8
    %     if i<=3
    subplot(8,2,2*i-1)
    vec_aux = vec2mat(x(1:qtd_entradas,num_salto_amostras*i),8);
    plot(vec_aux); ylim([-1 1])
    title(['Domínio do tempo, amostra:' num2str(num_salto_amostras*i)]);
    subplot(8,2,2*i)
    vec_aux = vec2mat(x(qtd_entradas+1:qtd_entradas+qtd_pontos_P1,num_salto_amostras*i),8);
    plot(f,vec_aux(1:n/2,:));
    title(['FFT, amostra: ' num2str(num_salto_amostras*i)]);
%     subplot(9,3,3*i)
%     plot(vec2mat(x(qtd_entradas+qtd_pontos_P1+1:length(x),num_salto_amostras*i),64)); %ylim([-1 1]);
%     title(['CrossCorrelação, amostra: ' num2str(num_salto_amostras*i)]);
end

% num_salto_amostras = n_padroes_entrada/4;
% figure();
% for i=1:4
%     subplot(4,2,2*i-1); plot(lag/frequencia_amostragem,x(:,num_salto_amostras*i),'k');
%     title(['CrossCorrelação, amostra:' num2str(num_salto_amostras*i)]);
%     subplot(4,2,2*i);  plot(lag/frequencia_amostragem,x(:,num_salto_amostras*i-num_salto_amostras/2),'k');
%     title(['CrossCorrelação, amostra:' num2str(num_salto_amostras*i-num_salto_amostras/2)]);
% end
%% ======================= Fim Plotagens ==================================

%% Parâmetros do treinamento
alfa = 0.5;
beta_momentum = 0.5;
ciclos = 0;
LMS_total = 0;
LMS_media = 1;
LMS_alvo = 10^-7;

%% Inicialização dos Pesos

% CAMADA ESCONDIDA_1
% inicialização pelo método de Nguyen-Widrow
beta_1 = 0.7*nthroot(neuronios_escondidos_1,qtd_entradas);  % fator de escala
w1 = rand(qtd_entradas+qtd_pontos_P1,neuronios_escondidos_1)-0.5;       % inicalização aleatória entre -0.5 e +0.5
for i=1:neuronios_escondidos_1
    raiz_1 = sqrt(sum(w1(:,i).^2));
    w1(:,i) = (beta_1/raiz_1)*w1(:,i);
end
b1 = 2*beta_1*rand(neuronios_escondidos_1,1)-beta_1;          % bias inicializado entre -beta e +beta

% CAMADA ESCONDIDA 2
beta_2 = 0.7*nthroot(neuronios_escondidos_2,neuronios_escondidos_1);  % fator de escala
w2 = rand(neuronios_escondidos_1,neuronios_escondidos_2)-0.5;       % inicalização aleatória entre -0.5 e +0.5
for i=1:neuronios_escondidos_2
    raiz_2 = sqrt(sum(w2(:,i).^2));
    w2(:,i) = (beta_2/raiz_2)*w2(:,i);
end
b2 = 2*beta_2*rand(neuronios_escondidos_2,1)-beta_2;          % bias inicializado entre -beta e +beta

% CAMADA DE SAÍDA
w3 = rand(neuronios_escondidos_2,neuronios_saida)-0.5;    % pesos aleatorios entre -0.5 e +0.5
b3 = rand(neuronios_saida,1)-0.5;

w1_anterior = zeros(qtd_entradas+qtd_pontos_P1,neuronios_escondidos_1);
b1_anterior = zeros(neuronios_escondidos_1,1);
w2_anterior = zeros(neuronios_escondidos_1,neuronios_escondidos_2);
b2_anterior = zeros(neuronios_escondidos_2,1);
w3_anterior = zeros(neuronios_escondidos_2,neuronios_saida);
b3_anterior = zeros(neuronios_saida,1);

%% FASE DE TREINAMENTO

u1 = zeros(1,neuronios_escondidos_1);     % somatória neurônios 1ª camada
y1 = zeros(1,neuronios_escondidos_1);     % sáida neurônios 1ª camada
u2 = zeros(1,neuronios_escondidos_2);     % somatória neurônios 2ª camada
y2 = zeros(1,neuronios_escondidos_2);     % sáida neurônios 2ª camada
u3 = zeros(1,neuronios_saida);            % somatória neurônios 3ª camada
y3 = zeros(1,neuronios_saida);            % sáida neurônios 3ª camada

delta1 = zeros(1,neuronios_escondidos_1);
delta2 = zeros(1,neuronios_escondidos_2);
delta3 = zeros(1,neuronios_saida);

while ((LMS_media > LMS_alvo)||(ciclos==500))
    LMS_total = 0;
    ciclos = ciclos+1;
    for entrada_treinamento=1:n_padroes_entrada
        
        %% Fase FOWARD (CÁLCULO DAS SAÍDAS DA REDE)
        
        % Cálculo das saídas da 1ª camada escondida
        for j=1:neuronios_escondidos_1
            u1(j) = x(:,entrada_treinamento)'*w1(:,j) + b1(j);
            y1(j) = (2/(1+exp(-u1(j))))-1; % Sigmóide bipolar
        end
        % Cálculo das saídas da 2ª camada escondida
        for j=1:neuronios_escondidos_2
            u2(j) = y1*w2(:,j) + b2(j);
            y2(j) = (2/(1+exp(-u2(j))))-1;
        end
        % Cálculo das saídas dos neurônios de saída
        for j=1:neuronios_saida
            u3(j) = y2*w3(:,j) + b3(j);
            y3(j) = (2/(1+exp(-u3(j))))-1;
        end
        
        %% Fase BACKWARD (RETROPROPAGAÇÃO DO ERRO)
        
        % Camada de Saída       ->  2ª Camada Escondida
        for j=1:neuronios_saida         % Calculando delta de cada neuronio da camada de saida (camada 3)
            delta3(j) = (t(entrada_treinamento,j) - y3(j))*0.5*(1+y3(j))*(1-y3(j));
        end
        % Ajuste dos pesos da camada de saída
        %         w3 = w3_anterior + (alfa*delta3'*y2)' + beta_momentum*(w3 - w3_anterior);
        w3 = w3_anterior + alfa*y2'*delta3 + beta_momentum*(w3 - w3_anterior);
        b3 = b3_anterior + alfa*delta3';
        % 2ª Camada Escondida	->	1ª Camada Escondida
        for j=1:neuronios_escondidos_2  % Calculando delta de cada neuronio da 2ª camada escondida (camada 2)
            delta2(j) = delta3*w3(j,:)'*0.5*(1+y2(j))*(1-y2(j));
        end
        % Ajuste dos pesos da 2ª camada escondida
        %         w2 = w2_anterior + (alfa*delta2'*y1)' + beta_momentum*(w2 - w2_anterior);
        w2 = w2_anterior + alfa*y1'*delta2 + beta_momentum*(w2 - w2_anterior);
        b2 = b2_anterior + alfa*delta2';
        % 1ª Camada Escondida   ->  Camada de Entrada
        for j=1:neuronios_escondidos_1    % Calculando delta de cada neuronio da 1ª camada escondida (camada 1)
            delta1(j) = delta2*w2(j,:)'*0.5*(1+y1(j))*(1-y1(j));
        end
        % Ajuste dos pesos da 1ª camada escondida
        %         w1 = w1_anterior + (alfa*delta1'*x(entradas_treinamento,:))'; % + beta_momentum*(w1 - w1_anterior);
        w1 = w1_anterior + alfa*x(:,entrada_treinamento)*delta1 + beta_momentum*(w1 - w1_anterior);
        b1 = b1_anterior + alfa*delta1';
        
        w1_anterior = w1;
        b1_anterior = b1;
        w2_anterior = w2;
        b2_anterior = b2;
        w3_anterior = w3;
        b3_anterior = b3;
        
        %% Cálculo do Erro Quadrático Médio Total
        
        for j=1:neuronios_saida
            LMS_total = LMS_total + 0.5*((t(entrada_treinamento,j) - y3(j))^2);
        end
        
    end
    LMS_media = (LMS_total/n_padroes_entrada)/neuronios_saida;
end

%% Gravação das matrizes de pesos já treinadas
save('w1','w1');
save('b1','b1');
save('w2','w2');

disp('Execução sem erros!');
save('b2','b2');
save('w3','w3');
save('b3','b3');