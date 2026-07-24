classdef test_rand_like< TestCase
    properties
        this_folder
        test_ref_f = 'rand_like_ref_sequence.mat'
        ref_seq
        old_matlab
    end
    methods
        function this=test_rand_like(varargin)
            if nargin==0
                name = 'test_rand_like';
            else
                name = varargin{1};
            end
            this = this@TestCase(name);
            this.this_folder = fileparts(which('test_rand_like.m'));
            ref_file = fullfile(this.this_folder,this.test_ref_f);
            old_matlab = verLessThan('MATLAB','9.13');
            % Matlab after 2021a calculates rand-like sequence differently,
            % so two types of results are available
            if is_file(ref_file)
                ld = load(ref_file);
                ref_struc = ld.ref_struc;
                if old_matlab
                    this.ref_seq = ref_struc.old_matlab_seq;
                else
                    this.ref_seq = ref_struc.new_matlab_seq;
                end
            else
                rand_like('start',42);
                ref_seq = rand_like([64*1024,1]);
                ref_struc = struct();
                if old_matlab
                    ref_struc.old_matlab_seq = ref_seq;
                else
                    ref_struc.new_matlab_seq = ref_seq;
                end
                this.ref_seq = ref_seq;
                save(ref_file,'ref_struc');
            end
            this.old_matlab = old_matlab;
        end

        function test_rand_like_consistency(this)
            rand_like('start',42);
            test_seq = rand_like([64*1024,1]);
            assertEqual(test_seq,this.ref_seq);
        end
        function test_consistency2(this)
            if ~this.old_matlab
                skipTest("Re #1960 fails in newer Matlab, so need to investigate if this test have any sense")
            end
            seeds_data= load(fullfile(this.this_folder,'sim_spe_testfun_seeds_file.mat'));
            seed1 = seeds_data.rnd_storage.seeds.gen_sqw_acc_sqw_spe_nomex1;
            seed2 = seeds_data.rnd_storage.seeds.gen_sqw_acc_sqw_spe_nomex1_fun;
            rand_like('start',seed1);
            seq1=rand_like([13921,1]);
            rand_like('start',seed2);
            seq2=rand_like([13921,1]);
            sample_file = fullfile(this.this_folder,'rand_like_ref_sequence2.mat');
            if is_file(sample_file)
                ref = load(sample_file);
                assertEqual(ref.seq1,seq1);
                assertEqual(ref.seq2,seq2);
            else
                save(sample_file,'seq1','seq2');
            end
        end

    end
end
