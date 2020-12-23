close all

% Veri k�mesine tabloyu y�kl�yoruz.
tbl = readtable('shortColorDataset.csv');

% T�m benzersiz renk adlar�n� yazd�r�yoruz.
unique(tbl.colorname)

% Benzersiz renk adlar�n� say�sal bir koda �evirme
[G,ID] = findgroups(tbl.colorname);


% Grupland�r�lm�� renk verilerini kullanarak tek bir kodlama olu�turma
target = ind2vec((G).');

% Sinir a��n�n e�itimi / testi i�in 3xN �zellikli bir matris olu�turma
x = [tbl.r.';tbl.g.';tbl.b.'];



% Sinir a��n� olusturma - patternnet burada bizim i�in i�in �o�unu yapar
net = patternnet(10);

%MSE = �|yi-F(xi)|'2 (regresyon a�lar� taraf�ndan al�nan hata 
%yaklas�m�n� en aza indirmeye cal�s�r)
%Croos Entropy =�pilogF(xi)(�apraz entropi)

% A�� ve x ve hedefleri kullanarak e�itimli a�� kullan�n.
net = train(net,x,target);

% G�rsel bir a� yap�s� ortaya ��karma i�iv
view(net)
y = net(x);

% performans kontrol�
perf = perform(net,target,y);
classes = vec2ind(y);

%% MATR�S ��Z�M�
accum_corr = [];
for ii = 1:length(ID)
    for jj = 1:length(ID)
        accum_corr(ii,jj) = sum( (classes == ii) & (G.' == jj));
    end
end

figure;
imagesc(accum_corr);
xticks([1:24]);
xticklabels( ID )
xtickangle(45);
yticks([1:24]);
yticklabels( ID )
title('Color Confusion Matrix', 'fontSize', 14);
set(gca, 'fontsize', 14);
