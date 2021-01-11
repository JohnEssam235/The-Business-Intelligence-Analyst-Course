import pandas as pd

# read the csv file
raw_csv_data = pd.read_csv('Absenteeism_data.csv')

#expand the columns and rows display
pd.set_option('display.max_rows', 700)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)

#print(raw_csv_data)

#hena howa 3ayz y3ml copy ll csv file da w y4t3'l 3ala el copy dy 34an mybwz4 el nos5a el aslya ....
# fa 3mlna copy ll raw_csv_data w 7tenaha f el data_frame
data_frame = raw_csv_data.copy()
#print(data_frame)

# get info of the data frame
#data_frame.info()


#.............................................................................................
# a7na 3ayzen n-predict eih ? el absenteeism from work .......
# fe el columns howa mdeek column asmo absenteeism in hours fa howa mdek kol employee bl ID bta3o w howa 3'ab kam sa3a youm kam...
# example : employee ID 26 3'aab 4 hours on data 7/7/2015...
# fa el absenteeism in hours dy to3tbr  "dependant variable" el howa el variable el leh 3laka bl 7aga el 3ayz a3mlha predict
# ba2y el columns asmohom "independent variables" dol mlhom4 3laka bl 7aga el 3ayz a3mlha predict bs hys3dony


#...........................................................................................
#fe columns m4 h7tagha f el analytics el h3mlo dlw2ty ... da homa hy2llo el precision aslun...zay eih ?
# el ID column da .... da ah by3rfny kol employee 3'aab kam sa3a aw b m3na asa7 men el employee el 3'ab 3adad sa3at mo3yna f youm mo3yn
# bs hal el ID da hyfedny f el analytics ???? la2 mlho4 lazma .. la2n ana 3ayz a7dd el absenteeism from work fa el ID mlho4 ay 3laka..
# fa h3mlo drop..


data_frame = data_frame.drop(['ID'],axis=1)  # axis = 1 ...m3naha enk btms7 el column el asmo ID .... lw axis = 0 kda htms7 el row
# mmkn t3ml bardo del data_frame['ID'] ...hya hya
# bs lw ht3ml drop mtnsa4 t3ml assign "=" ....

#print(data_frame)

#......................................................................................
#feh el column bta3 el reason for absence b numeric values...
# ana msln lw 3ayz a3raf as3'ar w akbar value h3ml kda

#print(data_frame['Reason for Absence'].min())
#print(data_frame['Reason for Absence'].max())

#lw 3ayz a3raf el unique values el fe el column da h3ml kda
#print(data_frame['Reason for Absence'].unique())

# lw 3ayz a3ml sort ll unique values dy h3ml kda
#print(sorted(data_frame['Reason for Absence'].unique()))  #hla2y en rakam 20 m4 mwgod msln..

# Note mohma awe ... el values f el column da numeric la2n el numeric values ashal b kter f el t3amol m3aha mn el text plus enha a7san
# fe el data storage la2nha bta5od memory a2l .....

# dlw2ty 34an a3ml quantitative analysis m7tag eny afsel el reason of absence column da l dummy values ( statistics ) ..h3ml kda
reason_columns = pd.get_dummies(data_frame['Reason for Absence']) # keda ana 3mlt data frame ll reason of absence lw7do
#print(reason_columns)

# ( statistics bardo ) 34an atgnb 7dos potenial multi collinearity issues.. dy 7aga kda statistics m4 fahmha bsra7a..lazm a4el el first column
# mn el reasons for absence data frame....

reason_columns = pd.get_dummies(data_frame['Reason for Absence'], drop_first = True)
#print(reason_columns)

# tb3an kda h4el el column bta3 el reason of absence mn el data frame la2n already fasaltha w 3mltha data frame lw7dha
data_frame = data_frame.drop(['Reason for Absence'], axis= 1)
#print(data_frame)


#.................................................................................
# h3ml 7aga asmha grouping ... da hysahel 3lya el analytics ... w howa eny h2sm el reasons for absence l groups ...
# kol group 3obara 3an data frame lw7do
# ana 2asemt el reason for absence l groups ... 4 groups ... kol group feh reasons mot4abha m3 b3d ....
# fa 3mlt data frame l kol group mn el 4 groups ...
#badal ma yekon 3andy 28 reason ... la2 ana 2asemthom l groups ... 4 groups ..... fa hykon sabab el absence howa wa7ed mn el groups dol bs..
# h3ml el 7war da ezay ??
#bst3ml functions asmha loc ... dy btgbly aw bt5leny a3ml access 3ala columns mo7dda...

reason_type1 = reason_columns.loc[:,1:14]  # awl group hyb2a feh columns mn 1 l 14
reason_type2 = reason_columns.loc[:,15:17] # tany group feh columns mn 15 l 17 .. and so on....
reason_type3 = reason_columns.loc[:,18:21]
reason_type4 = reason_columns.loc[:,22:28]
#print(reason_type1)
#print(reason_type2)

# tab ana 3ayz yb2a kol group represented by "one column" bs w "700 rows" .. el one column da b "1" lw awl group msln el 14 columns bto3o fehom 1
# w el one column da yb2a b "0" lw mfy4 wala column mn el 14 columns el fe awl group mfhom4 zero .. tab leh b3ml kda ??
# gm3t kol group f one column w 700 rows ... 34an a4of hal feh 7ad mn el employees 3'ab mn el sho3'l bsbb ay disease mn el group da ...
# badal ma yekon 3andy 28 reason ... la2 ana 2asemthom l groups ... 4 groups ..... fa hykon sabab el absence howa wa7ed mn el groups dol bs..
# h3ml el 7war da ezay ?? bl max(axis =1)
# shof el video latef ..............

reason_type1 = reason_columns.loc[:,1:14].max(axis = 1)  # kda da ydeny 1 column .. w value at each row mn el 700 ... lw el 14 columns el fe awl group da kan fehom 1 .. fa el value b 1 .. lw mfho4 1 fa yb2a b zero....
reason_type2 = reason_columns.loc[:,15:17].max(axis=1)
reason_type3 = reason_columns.loc[:,18:21].max(axis=1)
reason_type4 = reason_columns.loc[:,22:28].max(axis=1)
#print(reason_type1)
#print(reason_type2)


#................................................................................................
#dlw2ty b2a 3ayz a3ml concatenate ll data_frame w el reason_type1 w type2 w type3 w type4

data_frame = pd.concat([data_frame,reason_type1,reason_type2,reason_type3,reason_type4],axis=1)
#print(data_frame)

# fhmt b2a shwia kda ana b3ml eih ????
# ana el awl 2asemt el reason for absence l groups ... 4 groups ... kol group feh reasons mot4abha m3 b3d ....
# fa 3mlt data frame l kol group mn el 4 groups ...
# b3dha gm3t kol group f one column w 700 rows ... 34an a4of hal feh 7ad mn el employees 3'ab mn el sho3'l bsbb ay disease mn el group da ...
# badal ma yekon 3andy 28 reason ... la2 ana 2asemthom l groups ... 4 groups ..... fa hykon sabab el absence howa wa7ed mn el groups dol..
# w damagt el results bta3t el groups dy f el data_frame el aslya .........kda hysahel 3lya el analytics...



# 7aga baseta kda.............
# el columns bta3t el types esmha 0,1,2,3 .......fa 3ayz a3'yr asamehom...
# el data_frame.columns.values dy htgblk el names bta3t el columns kolha..5odhom copy paste w 7othom fe variable esmo column_names w 3'ayar
# b2a el names bta3t el columns el 3ayz t3'yrha..
#print(data_frame.columns.values)

column_names = ['Date','Transportation Expense','Distance to Work','Age',
 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children' ,'Pets',
 'Absenteeism Time in Hours','Reason_1', 'Reason_2','Reason_3','Reason_4']  #3'yrt asamehom ....

# b3dha 7ot asamy el columns b3d el ta3del fe el data frame bta3tk el aslya...............
data_frame.columns = column_names
#print(data_frame)

#...................................................................................
# Note 3ala ganb .... b nafs el technique el fo2 da bardo a2dr a3ml reordering ll colums
# print(data_frame.columns.values) w ha5od el values copy w paste w a7othom fe variable gded..w a3'yr fe el order bta3hom....

columns_reordered = ['Reason_1', 'Reason_2', 'Reason_3' ,'Reason_4','Date', 'Transportation Expense','Distance to Work' ,'Age',
 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets',
 'Absenteeism Time in Hours']   # reordered columns ...7othm fe el dataframe el aslya b2a

data_frame = data_frame[columns_reordered]

#print(data_frame)

#...................................................................................
# Note 3ala ganb ..... el .head() da bygblk awl 5 rows f el data frame bat3tk...
#print(data_frame.head())

#..................................................................................................................
# bl nsba ll date column h3ml 3leh shwia operations htfham ana leh 3mlt kda f el a5er .....................
# bs el awl h3ml 7aga asmha check point .. eih dy ??? h3ml copy ll data frame bta3ty el aslya w a7otha f data frame gdeda asmha df_reason_mod
# leh b3ml kda ??? enta 3mtn lazm t3ml kda kol fatra f el code ...34an lw feh mo4kla aw logical error f el code sho3'lak kolo myboz4...
# zay ma enta bt3ml save kol shwia f el code b3d ma btktb kaza satr .... 34an lw 7slt mo4kla aw el kahraba at3t el code myro74...
# fa ana kol shwia b3ml copy ll data frame el wsltlha w a7otha fe data frame gdeda............
# el data frame el gdeda asmha df_reason_mod..

# awl 7aga el date shaklo kda 07/07/2015 msln ana 3ayzo kda 07-07-2015 ( el howa shakl el timestamp)
# h3ml kda ezay ?? ..

df_reason_mod = data_frame.copy()   # awl 7aga a5dt copy mn el data frame el aslya 34an m4 teboz bs aho .............
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'])  # ast5dmt el to_datetime dy function bt7wly l timestamp....
#print(df_reason_mod['Date'])  # atb3 el date kda ... htla2eh timestamp....lw 3mlt print kda hydek date bs mn 3'er time...tab fen el time ??
# el time howa m4 byktbo ... tab lw 3ayz a3mlo access ??.....a3ml kda ... da hygblk awl row el date w el time....
#print(df_reason_mod['Date'][0])

#......................dlw2ty 3ayz afsel el days w el months ....................
# awl 7aga el months..................
# h3ml for loop telef 3ala el date column ... a7na 3arfen el date column feh date w time fa ana 3ayz el date mle4 da3wa bl time
# fa h-access element element fe el for loop dy w kol element h-access el month bta3o bs .......w h7ot el months dy f el list
months_list = []
for i in range(700):
    months_list.append(df_reason_mod['Date'][i].month)
#print(months_list)  # dy el list el feha el months ...7ot el months b2a f el data frame bt3tk...
df_reason_mod['months_value'] = months_list
#print(df_reason_mod)

# nafs el klam h3mlo 3ala el days......................
# h3ml function esmha get_day ...dy bta5od el date kamel w trg3lo el day
def get_day(date_value): # one parameter el howa el date
    return date_value.weekday() # .weekday() dy function btrg3 el day of the date ...

df_reason_mod['day of the week'] = df_reason_mod['Date'].apply(get_day) #kol element mwgod f el date column f el data frame bta3ty h-apply 3leh function asmha get_day el ana m3rfha fo2 w b3d ma apply h7ot el day da f el list esmha "day of the week" h7otha gowa el data frame bta3ty y3ny htb2a column gowaha.....
#print(df_reason_mod)

# eih b2a el ana 3mlto kol da f el date column ??
# ana aslun kont 3ayz afara2 el date column l days w months lw7dohom ...........
# 34an da hydeny accuracy w precision a7san b kter.....
# fa 34an a3ml kda lazm a7wl el date l timestamp b3dha afsel el date da l days w months..........

#...................................................................................................
#a5er column h3mlo pre-processing howa el education.......................
# lw 3mlna count ll education column y3ny shofna el values el feh w kol value mtkrr kam mara....
# hla2y en value 1 mtkrr 583 ..... w ba2y el values el hya 2,3,4 mtkrren 2olyl awl ....
#print(df_reason_mod['Education'].value_counts())
# fa m3na kda en a3'lab el employees el 3ndy fe educational state b 1 ...ayan kanet el 1 dy b2a ana m3rf4 ....
# fa 2alk 5las ana hfsel el education l 2 columns ...wa7d el value bta3to howa el 583 w el tany feh el count bta3 ba2y el values(2,,3,4) la2n kda kda homa 3ddhom 2olyl...
# tab h3ml kda ezay????
# h3mlhom f shakl map .... el nas el 3ndhom key = 1 el value bta3thom b 583
# w el nas ely el key bta3hom b 2,3,4 lehom value lw7dhom el howa 3ddhom kolhm m3 b3d
df_reason_mod['Education'] = df_reason_mod['Education'].map({1:0,2:1,3:1,4:1})

# garab b2a a3ml counts ll education kda b2a ...........htla2y howa el keys el mwgoda 0,1 bs... w el values bta3t el 0 hya 583 w el values bta3t el 1 hya mgmo3 el counts bta3(2,3,4)..........
#print(df_reason_mod['Education'].value_counts())


#................................................................................................
#a3ml checkpoint a5era b2a
df_preprocessed = df_reason_mod.copy()
#print(df_preprocessed)



