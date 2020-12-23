close all

% Veri kümesine tabloyu yüklüyoruz.
tbl = readtable('shortColorDataset.csv');

% Tüm benzersiz renk adlarýný yazdýrýyoruz.
unique(tbl.colorname)

% Benzersiz renk adlarýný sayýsal bir koda çevirme
[G,ID] = findgroups(tbl.colorname);


% Gruplandýrýlmýþ renk verilerini kullanarak tek bir kodlama oluþturma
target = ind2vec((G).');

% Sinir aðýnýn eðitimi / testi için 3xN özellikli bir matris oluþturma
x = [tbl.r.';tbl.g.';tbl.b.'];



% Sinir aðýný olusturma - patternnet burada bizim için iþin çoðunu yapar
net = patternnet(10);

%MSE = £|yi-F(xi)|'2 (regresyon aðlarý tarafýndan alýnan hata 
%yaklasýmýný en aza indirmeye calýsýr)
%Croos Entropy =£pilogF(xi)(çapraz entropi)

% Aðý ve x ve hedefleri kullanarak eðitimli aðý kullanýn.
net = train(net,x,target);

% Görsel bir að yapýsý ortaya çýkarma içiv
view(net)
y = net(x);

% performans kontrolü
perf = perform(net,target,y);
classes = vec2ind(y);

%% MATRÝS ÇÝZÝMÝ
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
