%% Specify Basic information of different groups     
group_dir = 'D:\lanzhou dataset\EEG_128channels_resting_lanzhou_2015\after_fit_mat';     
group_files = dir([group_dir, filesep, '*.mat']);  
for i=1:length(group_files)
    subj_fn = group_files(i).name;
    EEG = pop_importdata('setname',group_files(i).name(1:end-4),...
    'data',strcat(group_dir, filesep, subj_fn),...
    'srate',250,...
    'dataformat','matlab',...
    'nbchan',128,...
    'chanlocs','loc.ced');   
    %EEG = pop_resample( EEG, 500);   
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',45);   
    %EEG = pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1);    
    %EEG = pop_select( EEG, 'rmchannel',{'OZ','O2','HEOG','VEOG','TRIGGER'});   
    EEG = pop_saveset( EEG, 'filename',strcat(group_files(i).name(1:end-4), '.set'), 'filepath',strcat(group_dir, filesep, 'beta'));  
end
