
group1_dir = 'D:\lanzhou dataset\EEG_128channels_resting_lanzhou_2015\after_fit_mat';     
group1_dir2 = 'D:\lanzhou dataset\EEG_128channels_resting_lanzhou_2015\after_fit_mat\_ica'; 
group1_files = dir([group1_dir2, filesep, '*.set']);  
for i=1:length(group1_files)
    subj_fn = group1_files(i).name;
    EEG = pop_loadset('filename',strcat(subj_fn(1:end-4), '.set'), 'filepath', group1_dir2);
    EEG = pop_iclabel(EEG, 'default');
    EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]); 
    EEG = pop_subcomp( EEG, [], 0);   
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(group1_files(i).name(1:end-4), '.set'), 'filepath',strcat(group1_dir, filesep, '_rm_ica')); 
end