import SimpleITK as sitk
import numpy as np
import os

def function_create_border_by_origin(directory_series_2D, out_directory_series_2D):

    '''
    Arguments:
        directory_series_2D - directory local of images series (ex: 'path/') (format string)
        out_directory_series_2D - directory where the series of 
            images will be save (ex: 'path_series_resized/') (format string)
        
    Return:
        void (save serie image on the format dcm - SimpleITK)

    '''

    #Read the dicom image series
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(directory_series_2D)
    
    #Scrolling and resizing the slicers
    for series_index in range(len(seriesIDs)):

        print("Dentro da funcao 'function_create_border_by_origin', valor de series_index {}".format(series_index))
        dicom_names = reader.GetGDCMSeriesFileNames(directory_series_2D, seriesIDs[series_index])
        
        # print(dicom_names)
        # print(type(dicom_names))
        # print(len(dicom_names))

        #Getting the name of the slice
        change = str(dicom_names[0])

        #Getting position of last bar
        position_bar = change.find("/SS", 90, -6)
        
        #Getting the name of patient without the format
        name_patient = change[position_bar+1:-4]
        
        #Reading the serie (slice)
        reader.SetFileNames(dicom_names)
        image_2D = reader.Execute()
        size = image_2D.GetSize()

        #MUDAR AQUI (For create the slicers of the slicers_concatenated - .dcm in name_to_slice_resized)
        # For create the slicers of the growth map concatenated, you should rename name_to_slice_resized to .nii (technical arrangement for the error. kkk) 
        print("\nName of the file to be saved : {} \n".format(out_directory_series_2D + name_patient +".nii" ))
        print("Name of the patient: " + name_patient)
        name_to_slice_resized = name_patient +".nii" 
        #print( "Image size:", size[0], size[1], size[2] )

        #Getting values minimun and maximun of the image
        intensity_statistics_filter = sitk.StatisticsImageFilter()
        intensity_statistics_filter.Execute(image_2D)
        vmin_image_2D = intensity_statistics_filter.GetMinimum()
        
        vmax_image_2D= intensity_statistics_filter.GetMaximum()
        image_2D_array = sitk.GetArrayFromImage(image_2D)
        
        #Creating one static image and converting it to array
        image_larger_black = sitk.Image([256,256], sitk.sitkVectorUInt8,1)
        image_larger_black_array = sitk.GetArrayFromImage(image_larger_black)

        #Creating one image emply by of the array
        new_image_2D_array = np.empty(image_larger_black_array.shape)
        new_image_2D_array = new_image_2D_array.astype(np.uint8)

        #Attribuing the background of original image to new image (creating one image with background black)
        for r in range(new_image_2D_array.shape[0]): #rows
            for c in range(new_image_2D_array.shape[1]): #cols
                new_image_2D_array[r,c] = vmin_image_2D  

        #Getting the difference of the two slices (new slice and slice origin)
        add_rows = int( (new_image_2D_array.shape[0]/2) - (image_2D_array.shape[1]/2) )
        add_cols = int( (new_image_2D_array.shape[1]/2) - (image_2D_array.shape[2]/2) )
        print(add_rows)

        #Moving the image to center of the new slice
        for r in range(new_image_2D_array.shape[0]): #rows
            for c in range(new_image_2D_array.shape[1]): #cols
                 
                 if( (r < image_2D_array.shape[1]) and (c < image_2D_array.shape[2]) ):
                    new_image_2D_array[r + add_rows, c + add_cols] = image_2D_array[0,r,c]           

        #Converting the array of the new slice to image of the slice
        new_image_2D = sitk.GetImageFromArray(new_image_2D_array)

        #nome_do_mapa_crescimento = "Resultados/" + f[-6:] + "_mapa_crescimento_volume_resultado.nii"
        #Saving the new slice resized
        sitk.WriteImage(new_image_2D, name_to_slice_resized)

def function_create_border_by_origin_3D(name_volumn_3D, filename, path_raiz, extension):

    '''
    Arguments:
        directory_series_3D - directory local of images volumn (ex: 'path/') (format string)
        out_directory_series_3D - directory where the volumn of 
            images will be save (ex: 'path_series_resized/') (format string)
        
    Return:
        void (save serie image on the format dcm - SimpleITK)

    '''
    image_origin = sitk.ReadImage(name_volumn_3D)
    image_origin_array = sitk.GetArrayFromImage(image_origin)
    
    # #Creating one static image and converting it to array
    # image_larger_black = sitk.Image([image_origin_array[0],512,512], sitk.sitkVectorUInt8,1)
    # image_larger_black_array = sitk.GetArrayFromImage(image_larger_black)
    
    #Creating one image emply by of the array
    new_image_3D_array = np.empty((image_origin_array.shape[0],512,512))
    new_image_3D_array = new_image_3D_array.astype(np.uint8)
    new_image_3D_array = new_image_3D_array.reshape((image_origin_array.shape[0],512,512))

    #Attribuing the background of original image to new image (creating one image with background black)
    for s in range(new_image_3D_array.shape[0]): #slicers
        for r in range(new_image_3D_array.shape[1]): #rows
            for c in range(new_image_3D_array.shape[2]): #cols
                new_image_3D_array[s,r,c] = -32000  

    #Getting the difference of the two slices (new slice and slice origin)
    add_rows = int( (new_image_3D_array.shape[1]/2) - (image_origin_array.shape[1]/2) )
    add_cols = int( (new_image_3D_array.shape[2]/2) - (image_origin_array.shape[2]/2) )
    

    #Moving the image to center of the new slice
    for s in range(new_image_3D_array.shape[0]): #slicers
        for r in range(new_image_3D_array.shape[1]): #rows
            for c in range(new_image_3D_array.shape[2]): #cols
                    
                    if( (r < image_origin_array.shape[1]) and (c < image_origin_array.shape[2]) ):
                        new_image_3D_array[s, r + add_rows, c + add_cols] = image_origin_array[s,r,c]           

    #Converting the array of the new slice to image of the slice
    new_image_3D = sitk.GetImageFromArray(new_image_3D_array)

    #nome_do_mapa_crescimento = "Resultados/" + f[-6:] + "_mapa_crescimento_volume_resultado.nii"
    #Saving the new slice resized

    size_name_patient = len(filename[:6])
    sub_name_mod = filename[size_name_patient:] # path_name_patient - name_patient_without_extention
    name_to_slice_resized_nii = path_raiz + filename[:6] + "_resized" + sub_name_mod[:-4] + '.nii'
    name_to_slice_resized_mha = path_raiz + filename[:6] + "_resized" + sub_name_mod[:-4] + '.mha'
    
    name_final_file = name_to_slice_resized_mha
    if(len(filename)>14): # It means that the file is a mask (pattern of PLD base)
        name_final_file = name_to_slice_resized_nii
    sitk.WriteImage(new_image_3D, name_final_file)
    print("File " + filename + " terminado" )


path_raiz = input("Write the path of the volumns: ")

for filename in os.listdir(path_raiz):
       extension = filename[-4:]
       function_create_border_by_origin_3D(path_raiz + filename, filename, path_raiz, extension)
       



# path_raiz = input("Write the path of the volumns: ") 
 
# path_series_image = path_raiz + "Series/"
# try:  
#     os.mkdir(path_series_mask_temp_2_image)
# except OSError:  
#     print ("Creation of the directory of the mask temp 2 patient %s failed" % path_series_mask_temp_2_image)
# else:  
#     print ("Successfully created the directory f the mask temp 2 patient %s " % path_series_mask_temp_2_image)


# print("Working with the directory:  ", path_series_mask_temp_2_image)
# for z in range(imagem_mask_Temp_2_array.shape[0]):
#     name = path_series_mask_temp_2_image + f[-6:] + "_mask_fatia_" + str(z) + ".dcm" #Creating directory of series
#     sitk.WriteImage(mask_temp_2_volume[:,:,z],name)