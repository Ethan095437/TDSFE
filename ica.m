group1_dir = 'D:\lanzhou dataset\EEG_128channels_resting_lanzhou_2015\after_fit_mat';     
group1_dir1 = 'D:\lanzhou dataset\EEG_128channels_resting_lanzhou_2015\after_fit_mat\_preica';     
group1_files = dir([group1_dir1, filesep, '*.set']);  
for i=1:length(group1_files)
    subj_fn = group1_files(i).name;
    EEG = pop_loadset('filename',strcat(subj_fn(1:end-4), '.set'), 'filepath', strcat(group1_dir, filesep, '_preica')); 
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');   
    EEG = pop_saveset( EEG, 'filename',strcat(group1_files(i).name(1:end-4), '.set'), 'filepath',strcat(group1_dir, filesep, '_ica')); 
end
