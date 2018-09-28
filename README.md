# spatial_evolution
2D evolutionary spatial simulation of cells

Requirements:
    `scipy\\
    numpy
    matplotlib
    PIL
    ffmpeg`
    
installation:
    `git clone https://github.com/atarashansky/spatial_evolution.git
    cd spatial_evolution
    conda create env_name python=3.6
    conda activate env_name
    python setup.py install
    conda install -c menpo ffmpeg`

mandatory arguments:

    "input_image" -- path to the input image
    "dirname" -- path of directory to create (or not create if it already exists) where all data will be saved
    "name" -- name of this particular simulation run (files will be named with this)


optional arguments with default values:

    dt = 2400 (time step in seconds)
    numgen = 200000 (number of frames to run simulation for)
    make_movies = True (if True, makes movies. need ffmpeg installed for this)
    sb = 0.1, selection strength for beneficial mutations (10%)
    sd = 0.01, selection strength for deleterious
    mud = 0.1, mutation rate for deleterious
    mub = 1e-5, mutation rate for beneficial
    sampling = 250, save snapshot of simulation every 250 frames
    avg=1.16e-5, birth rate of cells per second (corresponds to ~24 hour birth rate)


The output data are:

    ims_fitness = a list of 2D numpy arrays with the fitness values of each cell in their respective locations
    ims_primarylineages = unique integer ID corresponding to a unique mutation. These mutations are the FIRST ones out of the homogeneous background (basically, the first mutation acquired for each cell).
    ims_extant = unique integer ID corresponding to a unique mutation. These mutations are the latest / extant mutations (basically, the lastest mutation acquired for each cell).
    "_muller.png" -- muller plot
    "_primary.mp4" -- movie of primary mutations
    "_extant.mp4" -- movie of extant mutations



