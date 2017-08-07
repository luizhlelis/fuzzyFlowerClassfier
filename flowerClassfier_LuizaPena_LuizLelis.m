clear
close all
clc

% **********************************************************
% ************ TRATAMENTO INICIAL DOS DADOS ****************
% **********************************************************

% CONJUNTO DE TREINEMNTO: x_train e y_train
x_train = reshape(textread('x_train.txt', '%f'),2,105);
x_train = x_train';
size_in = size(x_train);

x_lenght = x_train(:, 1);
x_width = x_train(:, 2);

% Verifica a faixa de valores das entradas e adiciona uma margem de erro
min_lenght = 0;
max_lenght = int64(max(x_lenght) + 0.5*max(x_lenght));
min_width = 0;
max_width = int64(max(x_width) + 0.2*max(x_width));

y_train = reshape(textread('y_train.txt', '%f'),1,105);
y_train = y_train';

% Apenas visualizacao do resultado esperado do CONJUNTO TREINAMENTO

% Y = 1 -> Iris Setosa (Blue)
% Y = 2 -> Iris Versicolor (Red)
% Y = 3 -> Iris Virginica (Green)

 figure(1)
 for i=1:size_in(1)
     if y_train(i) == 1
         plot(x_lenght(i),x_width(i),'b*');
         hold on;
     elseif y_train(i) == 2
         plot(x_lenght(i),x_width(i),'r*');
         hold on;
     else
         plot(x_lenght(i),x_width(i),'g*');
         hold on; 
     end 
 end
 title('Resultado esperado de classificacao das petalas - TREINAMENTO')
 set(gca,'XLim',[0 max_lenght])
 set(gca,'YLim',[0 max_width])

% CONJUNTO DE TESTE: x_test e y_test
x_test = reshape(textread('x_test.txt', '%f'),2,45);
x_test = x_test';
size_in_test = size(x_test);

x_lenght_test = x_test(:, 1);
x_width_test = x_test(:, 2);

y_test = reshape(textread('y_test.txt', '%f'),1,45);
y_test = y_test';

% Apenas visualizacao do resultado esperado do CONJUNTO TESTE

% Y = 1 -> Iris Setosa (Blue)
% Y = 2 -> Iris Versicolor (Red)
% Y = 3 -> Iris Virginica (Green)
figure(2)
for i=1:size_in_test(1)
    if y_test(i) == 1
        plot(x_lenght_test(i),x_width_test(i),'b*');
        hold on;
    elseif y_test(i) == 2
        plot(x_lenght_test(i),x_width_test(i),'r*');
        hold on;
    else
        plot(x_lenght_test(i),x_width_test(i),'g*');
        hold on; 
    end 
end
title('Resultado esperado de classificacao das petalas - TESTE')
set(gca,'XLim',[0 max_lenght])
set(gca,'YLim',[0 max_width])

fig = 3;

% **********************************************************
% ************** TREINAMENTO DO CLASSIFICADOR **************
% **********************************************************
% Define os Hiperparametros de entrada do classificador
MiMin = 1;
MiMax = 10;
for nMi=1:MiMax

    % Calcula espacamento uniforme para as funcoes de pertinencia
    
    lenght_spacing = (max_lenght - min_lenght)/(nMi-1);
    width_spacing = (max_width - min_width)/(nMi-1);

    % As funcoes de pertinencia serao triagulares.
    % A primeira funcao de pertinencia comecara com centro em zero e
    % as consequentes terao seu centro espacado conforme definido no
    % spacing calculado.

    % Sao calculadas as pertinencias para todas as entradas e, para cada 
    % entrada, eh reservada uma linha na matriz desse resultado, sendo cada
    % coluna dessa linha o resultado daquela entrada para a respectiva
    % funcao de pertinencia
    for x=1:size_in(1)
        for m=1:nMi
            lenght_a(m) = double((m-2)*lenght_spacing);
            lenght_b(m) = double((m-1)*lenght_spacing);
            lenght_c(m) = double((m)*lenght_spacing);
            mis_lenght(x, m) = trimf(x_lenght(x), [lenght_a(m) lenght_b(m) lenght_c(m)]);

            width_a(m) = double((m-2)*width_spacing);
            width_b(m) = double((m-1)*width_spacing);
            width_c(m) = double((m)*width_spacing);
            mis_width(x, m) = trimf(x_width(x), [width_a(m) width_b(m) width_c(m)]);
        end
    end

    % A Pertinencia do vetor de entrada sera o produto das funcoes de 
    % pertinencia de largura e altura.
    mx = 1;
    my = 1;
    k = 1;
    for i=1:size_in(1)
        k = 1;
        for my = 1:nMi
            for mx = 1:nMi
               mis(i, k) = mis_lenght(i,mx)*mis_width(i,my);
               k = k+1;
            end
        end 
    end

    % Classifica as nMi regioes entre flor 1, 2 ou 3
    beta = zeros (3,nMi^2);
    for j=1:(nMi^2)
        n1 = 0;
        n2 = 0;
        n3 = 0;
        for i=1:size_in(1)
            if (mis(i,j) ~= 0)
                if y_train(i) == 1
                    n1 = n1+1;
                    beta(1,j) = beta(1,j) + mis(i,j);
                elseif y_train(i) == 2
                    n2 = n2+1;
                    beta(2,j) = beta(2,j) + mis(i,j);
                else
                    n3 = n3+1;
                    beta(3,j) = beta(3,j) + mis(i,j);
                end
            end
        end
        [C,k] = max([n1,n2,n3]);
        if(C==0)
            Rk(j) = 0;
            w(j) = 0;
        elseif(k == 1)
            Rk(j) = 1;
            % Define o grau de certeza daquela regra
            w(j) = (beta(1,j) - beta(2,j) - beta(3,j)) / (beta(1,j) + beta(2,j) + beta(3,j));
        elseif(k == 2)
            Rk(j) = 2;
            % Define o grau de certeza daquela regra
            w(j) = (-beta(1,j) + beta(2,j) - beta(3,j)) / (beta(1,j) + beta(2,j) + beta(3,j));
        elseif(k == 3)
            Rk(j) = 3;
            % Define o grau de certeza daquela regra
            w(j) = (-beta(1,j) - beta(2,j) + beta(3,j)) / (beta(1,j) + beta(2,j) + beta(3,j));
        end
    end

    % *******************************************************
    % ********* TESTE DO CLASSIFICADOR - TREINAMENTO ********
    % *******************************************************

    % Rk eh rj, ou seja, os consequentes de cada regra
    % w eh o grau de certeza para uma determinada regra j

    % Primeiro, vamos testar com o conjunto de treinamento mesmo...

     erro_train = 0;
     for i=1:size_in(1)
         for m=1:nMi
             mis_lenght(i, m) = trimf(x_lenght(i), [lenght_a(m) lenght_b(m) lenght_c(m)]);
             mis_width(i, m) = trimf(x_width(i), [width_a(m) width_b(m) width_c(m)]);
         end
         k = 1;
         for my = 1:nMi
             for mx = 1:nMi
                mis1(i, k) = mis_lenght(i,mx)*mis_width(i,my);
                k = k+1;
             end
         end 
         for j = 1:(k-1)
             miXcerteza(j) = mis1(i, j)*w(j);
         end
         [mi_chapeu(i),R_chapeu(i)] = max(miXcerteza);
         y_chapeu(i) = Rk(R_chapeu(i));
         C_chapeu(i) = (R_chapeu(i));
         if(y_chapeu(i) ~= y_train(i))
             erro_train = erro_train+1;
         end
     end
     erro_train_100 = (erro_train/size_in(1))*100;
 
     n1 = 1;
     n2 = 1;
     n3 = 1;
     y_chapeu_plot_1 = zeros(1,2);
     y_chapeu_plot_2 = zeros(1,2);
     y_chapeu_plot_3 = zeros(1,2);
     for i=1:size_in(1)
         if y_chapeu(i) == 1
             y_chapeu_plot_1(n1,1) = x_train(i, 1);
             y_chapeu_plot_1(n1,2) = x_train(i, 2);
             n1 = n1 + 1;
         elseif y_chapeu(i) == 2
             y_chapeu_plot_2(n2,1) = x_train(i, 1);
             y_chapeu_plot_2(n2,2) = x_train(i, 2);
             n2 = n2 + 1;
         else
             y_chapeu_plot_3(n3,1) = x_train(i, 1);
             y_chapeu_plot_3(n3,2) = x_train(i, 2);
             n3 = n3 + 1;
         end 
     end
     figure(fig)
     fig = fig+1;
     plot(y_chapeu_plot_1(:,1),y_chapeu_plot_1(:,2),'.b');
     hold on;
     plot(y_chapeu_plot_2(:,1),y_chapeu_plot_2(:,2),'.r');
     hold on;
     plot(y_chapeu_plot_3(:,1),y_chapeu_plot_3(:,2),'.g');
     title('Resultado da classificacao das petalas - CONJUNTO TREINAMENTO')
     legend('Y = 1 -> Iris Setosa','Y = 2 -> Iris Versicolor','Y = 3 -> Iris Virginica');
 
     grid on
     set(gca,'XLim',[0 max_lenght])
     set(gca,'XTick',(0:(max_lenght - min_lenght)/(nMi):max_lenght))
     set(gca,'YLim',[0 max_width])
     set(gca,'YTick',(0:(max_width - min_width)/(nMi):max_width))

    % Agora, vamos realizar o mesmo procedimento para o conjunto de teste...

    % *******************************************************
    % *********** TESTE FINAL DO CLASSIFICADOR **************
    % *** VARIANDO O NUMERO DE SUBDIVISOES DO GRID FUZZY ****
    % *******************************************************

    % O vetor de plot_erro sera usado para printar a progressao do erro de
    % acordo com o numero de subdivisoes escolhido.

    erro_test = 0;
    for i=1:size_in_test(1)
        for m=1:nMi
            mis_lenght_test(i, m) = trimf(x_lenght_test(i), [lenght_a(m) lenght_b(m) lenght_c(m)]);
            mis_width_test(i, m) = trimf(x_width_test(i), [width_a(m) width_b(m) width_c(m)]);
        end
        k = 1;
        for my = 1:nMi
            for mx = 1:nMi
               mis_test(i, k) = mis_lenght_test(i,mx)*mis_width_test(i,my);
               k = k+1;
            end
        end 
        for j = 1:(k-1)
            miXcerteza_test(j) = mis_test(i, j)*w(j);
        end
        [mi_chapeu(i),R_chapeu(i)] = max(miXcerteza_test);
        y_chapeu_test(i) = Rk(R_chapeu(i));
        C_chapeu_test(i) = (R_chapeu(i));
        if(y_chapeu_test(i) ~= y_test(i))
            erro_test = erro_test+1;
        end
    end
    erro_test_100 = (erro_test/size_in_test(1))*100;
    plot_erro(nMi) = erro_test_100;

    n1 = 1;
    n2 = 1;
    n3 = 1;
    
    y_chapeu_plot_1_test = zeros(1,2);
    y_chapeu_plot_2_test = zeros(1,2);
    y_chapeu_plot_3_test = zeros(1,2);
    for i=1:size_in_test(1)
        if y_chapeu_test(i) == 1
            y_chapeu_plot_1_test(n1,1) = x_test(i, 1);
            y_chapeu_plot_1_test(n1,2) = x_test(i, 2);
            n1 = n1 + 1;
        elseif y_chapeu_test(i) == 2
            y_chapeu_plot_2_test(n2,1) = x_test(i, 1);
            y_chapeu_plot_2_test(n2,2) = x_test(i, 2);
            n2 = n2 + 1;
        else
            y_chapeu_plot_3_test(n3,1) = x_test(i, 1);
            y_chapeu_plot_3_test(n3,2) = x_test(i, 2);
            n3 = n3 + 1;
        end 
    end

    figure(fig)
    fig = fig+1;
    plot(y_chapeu_plot_1_test(:,1),y_chapeu_plot_1_test(:,2),'.b');
    hold on;
    plot(y_chapeu_plot_2_test(:,1),y_chapeu_plot_2_test(:,2),'.r');
    hold on;
    plot(y_chapeu_plot_3_test(:,1),y_chapeu_plot_3_test(:,2),'.g');
    title('Resultado da classificacao das petalas - CONJUNTO TESTE')
    legend('Y = 1 -> Iris Setosa','Y = 2 -> Iris Versicolor','Y = 3 -> Iris Virginica');

    grid on
    set(gca,'XLim',[0 max_lenght])
    set(gca,'XTick',(0:(max_lenght - min_lenght)/(nMi):max_lenght))
    set(gca,'YLim',[0 max_width])
    set(gca,'YTick',(0:(max_width - min_width)/(nMi):max_width))

end

% Grafico do erro
figure(fig)
plot(MiMin:MiMax,plot_erro,'-')
legend('nMi');
xlabel('nMi')
ylabel('Erro')
title('Grafico do erro em funcao do numero de funcoes de pertinencia nMi');