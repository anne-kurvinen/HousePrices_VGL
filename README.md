## Huspris Prediktion i Västra Götalands län

### Rapport:

Analys och prediktion av huspriser i dom kommunerna som tillhör Västra Götalands län
Denna rapport sammanfattar processen för att förbereda data, välja modell, utvärdera prestanda och
föreslå förbättringar för att förutsäga huspriser i Västra Götalands län baserat på tillgängligt dataset.

### 1. Dataförberedelse

Datasetet laddades in och genomgick flera steg för rening och förberedelse:

Kolumner togs bort: Onödiga kolumner som ad_id, date_published och coordenates togs bort.
Filtrering på hustyp: Endast hus (typology_HOUSE) behölls för analysen.
Namnbyte av kolumner: Kolumnerna döptes om till svenska namn för tydlighet: land_area_sqm till tomtyta,
living_area_sqm till boyta, number_rooms till rum, typology_HOUSE till hus, asking_price_sek till utgångspris
och sqm_price_sek till pris_sqm.
Extrahering och filtrering av kommun: Kolumnerna address och location slogs samman. Kommunnamnet extraherades
från kolumnen location, och datasetet filtrerades sedan för att endast inkludera kommuner inom Västra Götalands län
(baserat på en angiven lista).
Hantering av saknade värden: Rader med saknade värden eller värdet noll i tomtyta och boyta togs bort.
Rader med värdet mindre än 1 i rum togs också bort.
Hantering av extremvärden (Outliers): IQR-metoden (Interquartile Range) tillämpades för att identifiera
och ta bort extremvärden i de numeriska kolumnerna tomtyta, boyta, rum och utgångspris.
Imputering av saknade värden: Eventuella kvarvarande saknade numeriska värden fylldes i med medelvärdet
för respektive kolumn.
Feature Engineering: En interaktionskolumn, boyta_rum_interaktion, skapades genom att multiplicera boyta med rum.
One-Hot Encoding: Den kategoriska kolumnen municipality omvandlades till numeriska kolumner med one-hot encoding.
Skalning: Numeriska features skalades med StandardScaler för att normalisera deras värdeintervall.

Datasetet delades sedan upp i träningsdata (80%) och testdata (20%).

### 2. Modellval och träning

Flera regressionsmodeller testades för att förutsäga utgångspris:

Linjär Regression: En enkel och tolkningsbar modell som fungerar bra om det finns linjära samband i datat.
Random Forest Regressor: En ensemble-modell som bygger på flera beslutsträd och ofta presterar bra på komplexa dataset.
Gradient Boosting Regressor: Ytterligare en kraftfull ensemble-modell som sekventiellt bygger träd för att
korrigera fel från föregående träd.
Modellerna tränades på träningsdatan med standardinställningar för Random Forest och Gradient Boosting.

### 3. Utvärdering av modellprestanda

Modellerna utvärderades på testdatan med hjälp av följande regressionsmetriker:

Mean Squared Error (MSE): Mäter medelkvadratfelet mellan predikterade och faktiska värden.
Lägre värde indikerar bättre prestanda.
R-squared (R²): Mäter hur stor andel av variansen i den beroende variabeln som förklaras av modellen.
Ett värde närmare 1 indikerar bättre förklaringsförmåga.
Mean Absolute Error (MAE): Mäter medelvärdet av de absoluta skillnaderna mellan predikterade och
faktiska värden. Lägre värde indikerar bättre prestanda.
Prestandan för modellerna var som följer:

Modell MSE R-squared MAE
Linjär Regression 1.9396e+12 0.4890 980087.59
Random Forest 2.8845e+12 0.3694 1304926.90
Gradient Boosting 1.8192e+12 0.5207 1021044.00
Notera: Metrikvärdena kan variera något beroende på den exakta datafiltreringen och outlierhanteringen.

Baserat på dessa metriker presterade Gradient Boosting Regressor bäst med högst R-squared (0.5207) och lägst MSE.
Linjär Regression kom på andra plats, medan Random Forest presterade sämst.
En R-squared på runt 0.52 indikerar att modellerna förklarar cirka hälften av variationen i bostadspriserna.
De höga värdena för MSE och MAE (i miljonklassen) tyder på att modellerna har en betydande genomsnittlig
felmarginal i sina prisprediktioner.

### 4. Förbättringsförslag

För att förbättra modellernas prestanda kan följande åtgärder övervägas:

Mer avancerad Feature Engineering:
Slå upp fastigheterna i offentliga fastighetsdatakällor (exempelvis Lantmäteriet eller Booli).
Eller använda geokoordinaterna för att matcha mot öppna databaser som har byggnadsår kopplat till plats.
Skapa fler interaktionstermer mellan relevanta numeriska features.
Lägg till polynomfunktioner för att fånga icke-linjära samband.
Inkludera tidsbaserade features om det finns tidsstämplar i datat (t.ex. säsongsvariationer).
Utforska mer detaljerade geografiska features, t.ex. avstånd till centrum, skolor, kommunikationer etc.,
om sådan data finns tillgänglig.

Hyperparameteroptimering: Använd metoder som GridSearchCV eller RandomizedSearchCV för att finjustera hyperparametrarna
för Random Forest och Gradient Boosting modellerna. Detta kan avsevärt förbättra deras prestanda.

Utvärdera andra modeller: Testa andra regressionsmodeller som t.ex. XGBoost, LightGBM, Support Vector Regressor (SVR)
eller enklare neurala nätverk.

Cross-Validation: Använd k-fold cross-validation för en mer robust utvärdering av modellernas prestanda och för att
säkerställa att resultaten inte är beroende av en specifik tränings-/testuppdelning.
Mer sofistikerad Outlierhantering: Utforska alternativa metoder för outlierdetektion och hantering som kan vara mer
lämpliga för specifika features.

Datainsamling: Om möjligt, samla in mer relevant data som kan påverka bostadspriser, t.ex. fastighetens ålder, skick,
renoveringar, närhet till service, brottsstatistik i området, etc.
Genom att implementera dessa förbättringsförslag kan modellernas förmåga att korrekt förutsäga bostadspriser potentiellt ökas.
# HousePrices_VGL
