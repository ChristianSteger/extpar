!+ Fortran module to aggregate GLCC land use data to a target grid 
!
! History:
! Version      Date       Name
! ------------ ---------- ----
! V1_0         2010/12/21 Hermann Asensio
!  Initial release
! V1_1         2011/01/20 Hermann Asensio
!  small bug fixes accroding to Fortran compiler warnings
! V1_2         2011/03/25 Hermann Asensio
!  update to support ICON refinement grids
! V1_4         2011/04/21 Hermann Asensio
!  clean up
! V1_7         2013/01/25 Guenther Zaengl 
!   Parallel threads for ICON and COSMO using Open-MP, 
!   Several bug fixes and optimizations for ICON search algorithm, 
!   particularly for the special case of non-contiguous domains; 
!   simplified namelist control for ICON  
!
! Code Description:
! Language: Fortran 2003.
!=======================================================================
!> Fortran module to aggregate GLCC land use data to a target grid 
!!
!> \author Hermann Asensio
MODULE mo_agg_glcc

  USE mo_logging
  USE mo_kind,                  ONLY: wp, i4

  USE mo_grid_structures,       ONLY: igrid_icon, igrid_cosmo, & 
       &                              target_grid_def
                                
  USE mo_search_ll_grid,        ONLY: find_reg_lonlat_grid_element_index
                                
  USE mo_io_utilities,          ONLY: check_netcdf

  USE mo_search_target_grid,    ONLY: find_nearest_target_grid_element

  USE netcdf,                   ONLY :   &
       &                              nf90_open,              &
       &                              nf90_close,             &
       &                              nf90_inq_varid,         &
       &                              nf90_get_var,           &
       &                              NF90_NOWRITE

  USE mo_glcc_data,             ONLY: glcc_grid, &
       &                              lon_glcc,  &
       &                              lat_glcc

  USE mo_glcc_lookup_tables,    ONLY: name_lookup_table_glcc, & 
       &                              init_glcc_lookup_tables, get_name_glcc_lookup_tables, & 
       &                              z0_lt_glcc, lnz0_lt_glcc, plc_mn_lt_glcc, plc_mx_lt_glcc, & 
       &                              lai_mn_lt_glcc, lai_mx_lt_glcc, rd_lt_glcc, emiss_lt_glcc, rs_min_lt_glcc   

  USE mo_glcc_lookup_tables,    ONLY: glcc_look_up


  USE mo_math_constants,        ONLY: deg2rad
  USE mo_physical_constants,    ONLY: re

  USE mo_target_grid_data,      ONLY: lon_geo, & 
       &                              lat_geo, & 
       &                              search_res

  USE mo_utilities_extpar,      ONLY: check_input_file
  
  IMPLICIT NONE

  PRIVATE

  PUBLIC :: agg_glcc_data_to_target_grid

  REAL(KIND=wp), PARAMETER :: rs_min_undef=999. !< undefined value for minimal stomata resistance

  CONTAINS

  !> Subroutine to aggregate glcc data to the target grid
  SUBROUTINE agg_glcc_data_to_target_grid(glcc_file,ilookup_table_glcc,undefined,       &
       &                                        tg,                                         &
       &                                        nclass_glcc,                             &
       &                                        glcc_class_fraction, & 
       &                                        glcc_class_npixel, &
       &                                        glcc_tot_npixel,   &
       &                                        fr_land_glcc ,     &  
       &                                        ice_glcc,          &
       &                                        z0_glcc, &
       &                                        root_glcc, &
       &                                        plcov_mn_glcc, &
       &                                        plcov_mx_glcc, &
       &                                        lai_mn_glcc,   &
       &                                        lai_mx_glcc, &
       &                                        rs_min_glcc, &
       &                                        urban_glcc,  &
       &                                        for_d_glcc,  &
       &                                        for_e_glcc, &
       &                                        emissivity_glcc    )    

    TYPE(target_grid_def), INTENT(IN)        :: tg  !< structure with target grid description

    CHARACTER (LEN=*), INTENT(IN)            :: glcc_file(:)  !< filename glcc raw data

    INTEGER(KIND=i4), INTENT(IN)             :: ilookup_table_glcc, & 
         &                                      nclass_glcc

    REAL (KIND=wp), INTENT(IN)               :: undefined            !< undef value


    INTEGER (KIND=i4), INTENT(OUT)           :: glcc_class_npixel(:,:,:,:),& 
         &                                      glcc_tot_npixel(:,:,:)  

    REAL (KIND=wp), INTENT(OUT)              :: glcc_class_fraction(:,:,:,:), & 
         &                                      fr_land_glcc(:,:,:), &  !< fraction land due to glcc raw data
         &                                      ice_glcc(:,:,:), &      !< fraction of ice due to glcc raw data
         &                                      z0_glcc(:,:,:), &       !< roughness length due to glcc land use data
         &                                      root_glcc(:,:,:), &     !< root depth due to glcc land use data
         &                                      plcov_mx_glcc(:,:,:), & !< plant cover maximum due to glcc land use data
         &                                      plcov_mn_glcc(:,:,:), & !< plant cover minimum due to glcc land use data
         &                                      lai_mx_glcc(:,:,:), &   !< Leaf Area Index maximum due to glcc land use data
         &                                      lai_mn_glcc(:,:,:), &   !< Leaf Area Index minimum due to glcc land use data
         &                                      rs_min_glcc(:,:,:), &   !< minimal stomata resistance due to glcc land use data
         &                                      urban_glcc(:,:,:), &    !< urban fraction due to glcc land use data
         &                                      for_d_glcc(:,:,:), &    !< deciduous forest (fraction) due to glcc land use data
         &                                      for_e_glcc(:,:,:), &    !< evergreen forest (fraction) due to glcc land use data
         &                                      emissivity_glcc(:,:,:) !< longwave emissivity due to glcc land use da

    !local variables
    REAL (KIND=wp)                          :: default_real, & 
         &                                     a_weight(1:tg%ie,1:tg%je,1:tg%ke), &  !< area weight of all raw data pixels in target grid
         &                                     a_class(1:tg%ie,1:tg%je,1:tg%ke,1:nclass_glcc), &  
         &                                     apix, &       !< area of a raw data pixel
         &                                     apix_e, &       !< area of a raw data pixel at equator
         &                                     point_lon, point_lat, & 
         &                                     pland, &           !< land cover                      (-)
         &                                     pice, &            !< ice fraction                    (-)
         &                                     plnz0, &           !< logarithm of roughness length   (m)
         &                                     proot, &           !< root depth                      (m)
         &                                     pmn, &             !< minimal plant cover             (-)
         &                                     pmx, &             !< maximum plant cover             (-)
         &                                     plaimn, &          !< minimal leaf area index         (m**2/m**2)
         &                                     plaimx, &          !< maximum leaf area index         (m**2/m**2)
         &                                     purb, &            !< urbanisation                    (-)
         &                                     pfor_d, &          !< deciduous forest                (-)
         &                                     pfor_e, &          !< evergreen forest                (-)
         &                                     pemissivity, &     !< surface thermal emissivity      (-)
         &                                     prs_min, &         !< minimum stomata resistance      (s/m)
         &                                     hp, &  ! height of Prandtl-layer
         &                                     lnhp, & 
         &                                     pwz0, &  ! weighted summand for z0
         &                                     area_tot, &    ! total area
         &                                     area_land, &   ! area with land
         &                                     area_plcov, &  ! area covered with plants
         &                                     bound_north_cosmo, &  !< northern boundary for COSMO target domain
         &                                     bound_south_cosmo, &  !< southern boundary for COSMO target domain
         &                                     bound_west_cosmo, &   !< western  boundary for COSMO target domain
         &                                     bound_east_cosmo  !< eastern  boundary for COSMO target domain

    INTEGER (KIND=i4)                       :: undefined_integer, &  ! undef value
         &                                     l, &  ! counters
         &                                     i_col, j_row, &  ! counter
         &                                     i_lu, j_lu, & 
         &                                     ie, je, ke, &   ! indices for target grid elements
         &                                     start_cell_id, &  !< ID of starting cell for ICON search
         &                                     i1, i2, & 
         &                                     glcc_data_row(glcc_grid%nlon_reg), & 
         &                                     glcc_data_pixel(1:1,1:1), & 
         &                                     lu, &   ! land use class
         &                                     ncid_glcc, &                              !< netcdf unit file number
         &                                     varid_glcc, &                !< id of variable
         &                                     nlon, & 
         &                                     k_error      ! error return code
         
    INTEGER (KIND=i4), ALLOCATABLE          :: ie_vec(:), je_vec(:), ke_vec(:)  ! indices for target grid elements


    CHARACTER (LEN=80)                      :: varname  !< name of variable
    ! Some stuff for OpenMP parallelization
    INTEGER(KIND=i4)                        :: num_blocks, ib, il, blk_len, istartlon, iendlon, nlon_sub, ishift
    !$   INTEGER :: omp_get_max_threads, omp_get_thread_num, thread_id
    !$   INTEGER (i4), ALLOCATABLE :: start_cell_arr(:)

    CALL logging%info('Enter routine: agg_glcc_data_to_target_grid')

    apix_e  = re * re * deg2rad* ABS(glcc_grid%dlon_reg) * deg2rad * ABS(glcc_grid%dlat_reg) 
    ! area of GLCC raw data pixel at equator

    hp   = 30.0      ! height of Prandtl-layer
    lnhp = LOG(hp)

    default_real = 0.0
    undefined_integer= NINT(undefined)

    glcc_class_fraction = default_real
    glcc_class_npixel   = undefined_integer
    glcc_tot_npixel = undefined_integer

    a_weight = default_real
    a_class  = default_real

    SELECT CASE(tg%igrid_type)
      CASE(igrid_icon)  ! ICON GRID
        ke = 1
      CASE(igrid_cosmo)  ! COSMO GRID
        ke = 1
        bound_north_cosmo = MAXVAL(lat_geo) + 0.05_wp  ! add some "buffer"
        bound_north_cosmo = MIN(bound_north_cosmo,90.0_wp)
        bound_south_cosmo = MINVAL(lat_geo) - 0.05_wp  ! add some "buffer"
        bound_south_cosmo = MAX(bound_south_cosmo,-90.0_wp)

        bound_east_cosmo = MAXVAL(lon_geo) + 0.25_wp  ! add some "buffer"
        bound_east_cosmo = MIN(bound_east_cosmo,180.0_wp)
        bound_west_cosmo = MINVAL(lon_geo) - 0.25_wp  ! add some "buffer"
        bound_west_cosmo = MAX(bound_west_cosmo,-180.0_wp)
    END SELECT

    emissivity_glcc(:,:,:) = 0.0_wp
    
    ! init lookup tables
    CALL init_glcc_lookup_tables(nclass_glcc, &
         &      ilookup_table_glcc, &
         &      z0_lt_glcc,            &
         &      lnz0_lt_glcc,          &
         &      plc_mn_lt_glcc,        &
         &      plc_mx_lt_glcc,        &
         &      lai_mn_lt_glcc,        &
         &      lai_mx_lt_glcc,        &
         &      rd_lt_glcc,            &
         &      emiss_lt_glcc,         &
         &      rs_min_lt_glcc)

    CALL get_name_glcc_lookup_tables(ilookup_table_glcc, name_lookup_table_glcc)  

    ! open netcdf file
    CALL check_input_file(TRIM(glcc_file(1)),__FILE__,__LINE__)
    CALL check_netcdf( nf90_open(TRIM(glcc_file(1)),NF90_NOWRITE, ncid_glcc),__FILE__,__LINE__)

    varname = 'glccbyte' ! I know that the longitude coordinates for the GLCC data are stored in a variable called 'glccbyte'

    CALL check_netcdf( nf90_inq_varid(ncid_glcc, TRIM(varname), varid_glcc),__FILE__,__LINE__)
    nlon = glcc_grid%nlon_reg
    ALLOCATE(ie_vec(nlon),je_vec(nlon),ke_vec(nlon))
    ie_vec(:) = 0
    je_vec(:) = 0
    ke_vec(:) = 0
    start_cell_id = 1

    ! Determine start and end longitude of search
    istartlon = 1
    iendlon = glcc_grid%nlon_reg
    IF (tg%igrid_type == igrid_icon) THEN
      DO i_col = 1, glcc_grid%nlon_reg
        point_lon = lon_glcc(i_col)
        IF (point_lon < tg%minlon) istartlon = i_col + 1
        IF (point_lon > tg%maxlon) THEN
          iendlon = i_col - 1
          EXIT
        ENDIF
      ENDDO
    ELSE IF (tg%igrid_type == igrid_cosmo) THEN
      DO i_col = 1, glcc_grid%nlon_reg
        point_lon = lon_glcc(i_col)
        IF (point_lon < bound_west_cosmo) istartlon = i_col + 1
        IF (point_lon > bound_east_cosmo) THEN
          iendlon = i_col - 1
          EXIT
        ENDIF
      ENDDO
    ENDIF
    nlon_sub = iendlon - istartlon + 1

    num_blocks = 1
    !$   num_blocks = omp_get_max_threads()
    IF (MOD(nlon_sub,num_blocks)== 0) THEN
      blk_len = nlon_sub/num_blocks
    ELSE
      blk_len = nlon_sub/num_blocks + 1
    ENDIF
    !$   ALLOCATE(start_cell_arr(num_blocks))
    !$   start_cell_arr(:) = 1
    WRITE(message_text,*) 'nlon_sub: ',nlon_sub,' num_blocks: ',num_blocks, ' blk_len: ',blk_len
    CALL logging%info(message_text)

    CALL logging%info('Start loop over glcc rows...')
    ! loop over rows of GLCC dataset
    rows: DO j_row=1,glcc_grid%nlat_reg
      point_lat = lat_glcc(j_row)

      IF (tg%igrid_type == igrid_cosmo) THEN ! CASE COSMO grid, omit I/O from hard disk when outside target domain
        IF ((point_lat > bound_north_cosmo).OR.(point_lat < bound_south_cosmo) ) THEN ! raw data out of target grid
          CYCLE rows
        ENDIF
      ELSE IF (tg%igrid_type == igrid_icon) THEN 
        IF (point_lat > tg%maxlat .OR. point_lat < tg%minlat) THEN
          CYCLE rows
        ENDIF
      ENDIF ! grid type

      ! read in pixels
      CALL check_netcdf(nf90_get_var(ncid_glcc, varid_glcc,  glcc_data_row,  &
           &               start=(/1,j_row/),count=(/nlon,1/)),__FILE__,__LINE__)
      apix = apix_e * COS(point_lat * deg2rad) ! area of raw data pixel (in [m**2])
      ie_vec(istartlon:iendlon) = 0
      IF (tg%igrid_type /= igrid_icon) THEN
        je_vec(:) = 0
        ke_vec(:) = 0
      ENDIF

!$OMP PARALLEL DO PRIVATE(ib,il,i_col,i1,i2,ishift,point_lon,thread_id,start_cell_id)
      DO ib = 1, num_blocks

        !$     thread_id = omp_get_thread_num()+1
        !$     start_cell_id = start_cell_arr(thread_id)
        ishift = istartlon-1+(ib-1)*blk_len

        columns1: DO il = 1,blk_len
          i_col = ishift+il
          IF (i_col > iendlon) CYCLE columns1

          ! find the corresponding target grid indices
          point_lon = lon_glcc(i_col)

          ! Reset start cell when entering a new row or when the previous data point was outside
          ! the model domain
          IF (tg%igrid_type == igrid_icon .AND. (il == 1 .OR. start_cell_id == 0)) THEN
            i1 = NINT(point_lon*search_res)
            i2 = NINT(point_lat*search_res)
            start_cell_id = tg%search_index(i1,i2)
            IF (start_cell_id == 0) EXIT ! in this case, the whole row is empty; may happen with merged (non-contiguous) domains
          ENDIF

          CALL  find_nearest_target_grid_element( point_lon,     &
               &      point_lat,     &
               &      tg,            &
               &      start_cell_id, &
               &      ie_vec(i_col), &
               &      je_vec(i_col), &
               &      ke_vec(i_col)  )

        ENDDO columns1
        !$     start_cell_arr(thread_id) = start_cell_id
      ENDDO
!$OMP END PARALLEL DO

      columns2: DO i_col=istartlon,iendlon

        ie = ie_vec(i_col)
        je = je_vec(i_col)
        ke = ke_vec(i_col)

        IF ((ie /= 0).AND.(je/=0).AND.(ke/=0))THEN 
          lu = glcc_data_row(i_col)                        ! land use class

          CALL glcc_look_up(lu, &
               &      nclass_glcc, &
               &      lnz0_lt_glcc,          &
               &      plc_mn_lt_glcc,        &
               &      plc_mx_lt_glcc,        &
               &      lai_mn_lt_glcc,        &
               &      lai_mx_lt_glcc,        &
               &      rd_lt_glcc,            &
               &      emiss_lt_glcc,         &
               &      rs_min_lt_glcc,        &
               &      pland,          &
               &      pice,           &
               &      plnz0,          &
               &      proot,          &
               &      pmn,            &
               &      pmx,            &
               &      plaimn,         &
               &      plaimx,         &
               &      purb,           &
               &      pfor_d,         &
               &      pfor_e,         &
               &      pemissivity,    &
               &      prs_min,        &
               &      k_error)


          IF (k_error == 0) THEN ! valid land use class

            ! raw data pixel within target grid, see output of routine find_rotated_lonlat_grid_element_index
            !- summation of variables
            glcc_tot_npixel(ie,je,ke) = glcc_tot_npixel(ie,je,ke) + 1
            a_weight(ie,je,ke) = a_weight(ie,je,ke) + apix  ! sum up area for weight

            glcc_class_npixel(ie,je,ke,lu) = glcc_class_npixel(ie,je,ke,lu) + 1
            a_class(ie,je,ke,lu) = a_class(ie,je,ke,lu) + apix   ! sum area of valid land use pixels 
            !(use as weight later)
            emissivity_glcc(ie,je,ke) =  emissivity_glcc(ie,je,ke) + apix * pemissivity
            IF (pland >  0.0) THEN ! only for land pixel

              ! weighted with whole area
              fr_land_glcc(ie,je,ke) = fr_land_glcc(ie,je,ke) + apix * pland
              ice_glcc(ie,je,ke) = ice_glcc(ie,je,ke) + apix * pice
              urban_glcc(ie,je,ke) = urban_glcc(ie,je,ke) + apix * purb
              ! z0 is averaged logarithmic
              IF ( lnhp /= plnz0) THEN ! z0 is averaged logarithmic
                pwz0 = 1./(lnhp - plnz0)
              ELSE
                pwz0 = 0.
              ENDIF
              z0_glcc(ie,je,ke)      = z0_glcc(ie,je,ke) + apix * pwz0
              plcov_mn_glcc(ie,je,ke) = plcov_mn_glcc(ie,je,ke) + apix * pmn
              plcov_mx_glcc(ie,je,ke) = plcov_mx_glcc(ie,je,ke) + apix * pmx

              ! the following fields are weighted with the plant cover
              root_glcc(ie,je,ke) = root_glcc(ie,je,ke) + apix * pmx * proot
              lai_mn_glcc(ie,je,ke) = lai_mn_glcc(ie,je,ke) + apix * pmx * plaimn
              lai_mx_glcc(ie,je,ke) = lai_mx_glcc(ie,je,ke) + apix * pmx * plaimx
              rs_min_glcc(ie,je,ke) = rs_min_glcc(ie,je,ke) + apix * pmx * prs_min
              for_d_glcc(ie,je,ke) = for_d_glcc(ie,je,ke) + apix * pmx * pfor_d
              for_e_glcc(ie,je,ke) = for_e_glcc(ie,je,ke) + apix * pmx * pfor_e

            END IF
          ENDIF
        ENDIF

        ! end loops
      ENDDO columns2
    ENDDO rows

    DEALLOCATE(ie_vec,je_vec,ke_vec)
    !$   DEALLOCATE(start_cell_arr)
    CALL logging%info('...done')

    ! calculate glcc_class_fraction (glcc_class_fraction/glcc_class_npixel)
    DO ke=1, tg%ke
      DO je=1, tg%je
        DO ie=1, tg%ie
          area_tot = a_weight(ie,je,ke)
          area_land = fr_land_glcc(ie,je,ke)

          ! weight by total area
          IF (area_tot > 0.0) THEN
            emissivity_glcc(ie,je,ke) = emissivity_glcc(ie,je,ke) / area_tot
            DO l=1,nclass_glcc
              glcc_class_fraction(ie,je,ke,l) =  a_class(ie,je,ke,l) /  area_tot 
              ! area fraction of each land use class
            ENDDO
          ENDIF

          IF (area_land > 0.0 ) THEN
            area_plcov = plcov_mx_glcc(ie,je,ke)
            !area_plcov = MAX(area_plcov, eps)      ! area covered with plants

            z0_glcc(ie,je,ke) = z0_glcc(ie,je,ke) / area_land
            z0_glcc(ie,je,ke) = hp * EXP(-1./z0_glcc(ie,je,ke))

            ! weight by total area
            fr_land_glcc(ie,je,ke) = fr_land_glcc(ie,je,ke) / area_tot
            ice_glcc(ie,je,ke)   = ice_glcc(ie,je,ke)   / area_tot


            ! weight by land area
            urban_glcc(ie,je,ke) = urban_glcc(ie,je,ke) / area_land
            for_d_glcc(ie,je,ke) = for_d_glcc(ie,je,ke) / area_land
            for_e_glcc(ie,je,ke) = for_e_glcc(ie,je,ke) / area_land
            plcov_mn_glcc(ie,je,ke) = plcov_mn_glcc(ie,je,ke) / area_land

            plcov_mx_glcc(ie,je,ke) = plcov_mx_glcc(ie,je,ke) / area_land

            ! weight by area covered with plants

            IF (area_plcov > 0.0) THEN 
              root_glcc(ie,je,ke) = root_glcc(ie,je,ke)     / area_plcov
              lai_mn_glcc(ie,je,ke) = lai_mn_glcc(ie,je,ke) / area_plcov
              lai_mx_glcc(ie,je,ke) = lai_mx_glcc(ie,je,ke) / area_plcov
              rs_min_glcc(ie,je,ke) = rs_min_glcc(ie,je,ke) / area_plcov
            ELSE ! may occur for ice grid elements
              root_glcc(ie,je,ke) = undefined
              lai_mn_glcc(ie,je,ke) = undefined
              lai_mx_glcc(ie,je,ke) = undefined
              rs_min_glcc(ie,je,ke) = rs_min_undef
            ENDIF


          ELSE IF ( area_tot > 0.0) THEN ! only sea pixels were found
            fr_land_glcc(ie,je,ke) = undefined
            z0_glcc(ie,je,ke)      = undefined
            ice_glcc(ie,je,ke)     = undefined
            urban_glcc(ie,je,ke)   = undefined
            for_d_glcc(ie,je,ke)   = undefined
            for_e_glcc(ie,je,ke)   = undefined
            plcov_mx_glcc(ie,je,ke)= undefined
            plcov_mn_glcc(ie,je,ke)= undefined

            root_glcc(ie,je,ke)    = undefined
            lai_mx_glcc(ie,je,ke)  = undefined
            lai_mn_glcc(ie,je,ke)  = undefined
            rs_min_glcc(ie,je,ke)  = undefined

          ENDIF
        ENDDO
      ENDDO
    ENDDO

    DO ke=1, tg%ke
      DO je=1, tg%je
        DO ie=1, tg%ie
          IF (glcc_tot_npixel(ie,je,ke) == 0) THEN ! nearest neighbor search

            point_lon = lon_geo(ie,je,ke)
            point_lat = lat_geo(ie,je,ke)
            CALL find_reg_lonlat_grid_element_index(point_lon,      &
                 &                                     point_lat,      &
                 &                                     glcc_grid,  &
                 &                                     i_lu,    &
                 &                                     j_lu)
            ! get data
            i_col = i_lu
            j_row = j_lu
            CALL check_netcdf(nf90_get_var(ncid_glcc, varid_glcc,  glcc_data_pixel,  &
                 &               start=(/ i_col,j_row /),count=(/ 1,1 /)),__FILE__,__LINE__)

            lu = glcc_data_pixel(1,1)


            CALL glcc_look_up(lu, &
                 &      nclass_glcc, &
                 &      lnz0_lt_glcc,          &
                 &      plc_mn_lt_glcc,        &
                 &      plc_mx_lt_glcc,        &
                 &      lai_mn_lt_glcc,        &
                 &      lai_mx_lt_glcc,        &
                 &      rd_lt_glcc,            &
                 &      emiss_lt_glcc,         &
                 &      rs_min_lt_glcc,        &
                 &      pland,          &
                 &      pice,           &
                 &      plnz0,          &
                 &      proot,          &
                 &      pmn,            &
                 &      pmx,            &
                 &      plaimn,         &
                 &      plaimx,         &
                 &      purb,           &
                 &      pfor_d,         &
                 &      pfor_e,         &
                 &      pemissivity,    &
                 &      prs_min,        &
                 &      k_error)


            IF (k_error == 0) THEN ! valid land use class
              apix = apix_e * COS(point_lat * deg2rad) ! area of raw data pixel (in [m**2])
              glcc_class_npixel(ie,je,ke,lu) = glcc_class_npixel(ie,je,ke,lu) + 1
              a_class(ie,je,ke,lu) = a_class(ie,je,ke,lu) + apix   ! sum area of valid land use pixels 
              emissivity_glcc(ie,je,ke) =  pemissivity
              IF (pland >  0.0) THEN ! only for land pixel
                fr_land_glcc(ie,je,ke) = pland
                ice_glcc(ie,je,ke) =  pice
                urban_glcc(ie,je,ke) = purb
                IF ( lnhp /= plnz0) THEN ! log z0 
                  pwz0 = 1./(lnhp - plnz0)
                ELSE
                  pwz0 = 0.
                ENDIF
                z0_glcc(ie,je,ke)      =  pwz0
                z0_glcc(ie,je,ke) = hp * EXP(-1./z0_glcc(ie,je,ke))
                plcov_mn_glcc(ie,je,ke) = pmn
                plcov_mx_glcc(ie,je,ke) = pmx
                root_glcc(ie,je,ke) = proot
                lai_mn_glcc(ie,je,ke) = plaimn
                lai_mx_glcc(ie,je,ke) = plaimx
                rs_min_glcc(ie,je,ke) = prs_min
                for_d_glcc(ie,je,ke) = pfor_d * pmx
                for_e_glcc(ie,je,ke) = pfor_e * pmx
                IF (pice == 1.0) THEN  ! ice land use class
                  root_glcc(ie,je,ke) = undefined
                  lai_mn_glcc(ie,je,ke) = undefined
                  lai_mx_glcc(ie,je,ke) = undefined
                  rs_min_glcc(ie,je,ke) = rs_min_undef
                ENDIF
              ENDIF
            ELSE ! not a valid land use class
              fr_land_glcc(ie,je,ke) = undefined
              z0_glcc(ie,je,ke)      = undefined
              ice_glcc(ie,je,ke)     = undefined
              urban_glcc(ie,je,ke)   = undefined
              for_d_glcc(ie,je,ke)   = undefined
              for_e_glcc(ie,je,ke)   = undefined
              plcov_mx_glcc(ie,je,ke)= undefined
              plcov_mn_glcc(ie,je,ke)= undefined
              root_glcc(ie,je,ke)    = undefined
              lai_mx_glcc(ie,je,ke)  = undefined
              lai_mn_glcc(ie,je,ke)  = undefined
              rs_min_glcc(ie,je,ke)  = undefined
            ENDIF
          ENDIF ! nearest neighbour search
        ENDDO
      ENDDO
    ENDDO

    ! close netcdf file 
    CALL check_netcdf( nf90_close(ncid_glcc),__FILE__,__LINE__)

    CALL logging%info('Exit routine: agg_glcc_data_to_target_grid')

  END SUBROUTINE agg_glcc_data_to_target_grid

END MODULE mo_agg_glcc
