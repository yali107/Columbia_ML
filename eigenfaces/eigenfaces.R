#############################
# This project is to explore PCA concepts using a dataset of Yale student face images. 
#############################

#################
# Setup
#################

library(pixmap)

pic_list <- 1:38
view_list <- c('P00A+000E+00', 'P00A+005E+10', 'P00A+005E-10', 'P00A+010E+00')
dir_list_1 <- dir(path="CroppedYale/",all.files=FALSE)

# preallocate an empty list
pic_data <- vector("list",length(pic_list)*length(view_list))
# preallocate an empty list to store the pgm for debugging
pic_data_pgm <- vector("list",length(pic_list)*length(view_list))

# outer loop through the pictures 
for ( i in 1:length(pic_list) ){
	# inner loop over views
	for ( j in 1:length(view_list) ){
		# compile the correct file name
		this_filename <- sprintf("CroppedYale/%s/%s_%s.pgm", dir_list_1[pic_list[i]] , dir_list_1[pic_list[i]] , view_list[j])

		# load the data
		this_face <- read.pnm(file=this_filename)
		this_face_matrix <- getChannels(this_face)
		if((i==1)&&(j==1)){ # Preallocate matrix to store picture vectors, store sizes for computations
			original_size <- dim(this_face_matrix)
			pic_vector_length <- prod(original_size)
			pic_mat <- mat.or.vec(length(pic_list)*length(view_list),pic_vector_length)
		}
		# store pgm as element of the list
		pic_data_pgm[[(i-1)*length(view_list)+j]] <- this_face
		# store matrix as element of the list
		pic_data[[(i-1)*length(view_list)+j]] <- this_face_matrix
		# make the face into a vector and include in the data matrix
		pic_mat[(i-1)*length(view_list)+j,] <-  as.vector(this_face_matrix)

	}	
}

# Find the total number of images
pic_mat_size <- dim(pic_mat)


#################
# Centralize the data by calculating and subtracting 
# the mean face of the data from each image.
#################

# Find the mean face vector
mean_face <- colMeans(pic_mat)
# Now print it as a picture
mean_face_mat <- mean_face
dim(mean_face_mat) <- original_size
mean_face_pix <- pixmapGrey(mean_face_mat)
plot(mean_face_pix)
# And save the picture
filename <- 'hw02_02b.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

# Subtract off the mean face
pic_mat_centered <- mat.or.vec(pic_mat_size[1],pic_mat_size[2])
# Using a loop to centralize all the images by subtract the mean face image from each image
for (i in 1:pic_mat_size[1]){
	pic_mat_centered[i,] <- pic_mat[i,] - mean_face
}


#################
# Apply principle component analysis and plot the explained variance
#################

pic_pca <- prcomp(pic_mat_centered)

# calculate the explained variance of each components and 
# make a vector to store the variances captured by the components
n_comp <- length(pic_pca$x[,1])
pca_var <- mat.or.vec(n_comp,1)
for (i in 1:n_comp){
	if (i==1){
		pca_var[i] <- pic_pca$sdev[i]^2
	}else{
		pca_var[i] <- pca_var[i-1] + pic_pca$sdev[i]^2
	}
}

pca_var <- pca_var/pca_var[n_comp]*100
# plot the explained variance vs all the components
plot(pca_var,ylim=c(-2,102),xlab="Number of Components",ylab="Percentage of Variance Captured")
# add a line at 100 to show max possible variance
abline(h=100,col="red")
# save the plot
filename <- 'hw02_02c.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

#################
# Plot the first 9 principle eigenfaces, there are the 
# top 9 images that explains the most variance
#################
eigenface_mat <- vector()
max_faces <- 9

# loop through first 9 eigenfaces
this_face_row <- vector()

for (i in 1:max_faces){
	# Make the eigenface vector into a matrix
	eigenface_temp <- pic_pca$rotation[,i]
	dim(eigenface_temp) <- original_size
	this_face_row <- cbind(this_face_row,eigenface_temp)
	if ((i %% 3)==0){
		# make a new row
		eigenface_mat <- rbind(eigenface_mat,this_face_row)
		# clear row vector
		this_face_row <- vector()
	}
}
# Plot the eigenfaces
eigenface_pgm <- pixmapGrey((eigenface_mat-min(eigenface_mat))/(max(eigenface_mat)-min(eigenface_mat)))
plot(eigenface_pgm)
# And save the plot
filename <- 'hw02_02d.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

#################
# Add the first 24 and 125 eigenfaces to the mean face 
# one at time to reconstruct the original face image
#################
# Find the index of face yaleB01_P00A+010E+00.pgm
face_index <- 20

# Initialize the function to produce a 5 x 5 matrix of faces
eigenface_add_by_face <- function(face_index, max_faces, by_faces, mean_face, pic_pca, original_size){
	# Initialize matrix for faces added in one eigenface at a time
	face_by_eig_mat <- vector()
	face_by_eig_row <- vector()
	face_by_eig_vector <- mean_face
	# Store the temporary face as a matrix
	face_temp <- face_by_eig_vector
	dim(face_temp) <- original_size
	face_by_eig_row <- cbind(face_by_eig_row,face_temp)
	
	# Now add in the eigenfaces
	for (i in 1:24){
		# Find the indices of the eigenfaces to include
		ind_include <- seq((i-1)*by_faces+1,i*by_faces,1)
		# Add up the vector that is score[j] x eigenface[j]
		eigenface_add <- mat.or.vec(length(mean_face),1)
		for (j in 1:length(ind_include)){
			ind_temp <- ind_include[j]
			eigenface_add <- eigenface_add + pic_pca$x[face_index,ind_temp]*pic_pca$rotation[,ind_temp]
		}
		face_by_eig_vector <- face_by_eig_vector + eigenface_add
		# Transform this back to matrix and include
		face_temp <- face_by_eig_vector
		dim(face_temp) <- original_size
		face_by_eig_row <- cbind(face_by_eig_row,face_temp)
		if ((i %% 5) == 4){
			# Start a new row
			face_by_eig_mat <- rbind(face_by_eig_mat,face_by_eig_row)
			face_by_eig_row <- vector()
		}
	}
	# Return the matrix of faces
	return(face_by_eig_mat) 
}

# Loop through the first 24 eigenfaces
max_faces <- 24
by_faces <- 1
face_by_1 <- eigenface_add_by_face(face_index, max_faces, by_faces, mean_face, pic_pca, original_size)
face_by_1_pm <- pixmapGrey(face_by_1)

# Plot and save results
plot(face_by_1_pm)
filename <- 'hw02_02e_1.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

# Loop through the first 120 eigenfaces
max_faces <- 120
by_faces <- 5
face_by_5 <- eigenface_add_by_face(face_index, max_faces, by_faces, mean_face, pic_pca, original_size)
face_by_5_pm <- pixmapGrey(face_by_5)

# Plot and save results
plot(face_by_5_pm)
filename <- 'hw02_02e_2.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

# Although the 120 eiganfaces approaches the original image more than the 24 eiganfaces, 
# but the difference is hardly distinguishable to the naked eye, therefore the 24 eiganfaces
# can sufficiently explain most of the variance.

