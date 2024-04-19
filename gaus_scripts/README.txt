December 14, 2021

This is a description of the seamount analysis:

01_model_all.sh - construct models for 35400 seamounts and height range 700 - 2600

02_gather_best.sh - this finds the best fit starting with lists of good seamounts
  24129 best_KW.txt
  10796 best_NEW.txt
  34925 best_all.txt

03_analyze_smt.m - this makes Figure 7 in the paper from the three best lists

04_analize_all.sh - this compiles all the relevant data for each seamount 
                    lon, lat, height, radius, base_depth, name
   35400 all.xyhrdn

045_sort_all.sh - this sorts the data into tall short shallow and good

   19523 good.xyhrdn
     572 shallow.xyhrdn
     532 short.xyhrdn
    2342 tall.xyhrdn

05_cm_all.sh - makes the xyz data for a list of seamounts
 
06_kml_all.sh - make kml files from *.xyn files

07_extract_739.sh - adds the measured and gravity_predicted summit depth to the special 739 seamounts

08_charted_uncharted.sh - finds the shallow charted and uncharted seamounts

09_uncharted_test.sh - not sure what this does

