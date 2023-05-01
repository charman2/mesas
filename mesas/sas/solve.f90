! -*- f90 -*-
subroutine solveSAS(J_ts, Q_ts, SAS_args, P_list, weights_ts, sT_init_ts, dt, &
                    verbose, debug, warning, jacobian, &
                    mT_init_ts, C_J_ts, alpha_ts, k1_ts, C_eq_ts, C_old, &
                    n_substeps, component_type, numcomponent_list, numargs_list, numflux, numsol, max_age, &
                    timeseries_length, index_ts, len_index_ts, numcomponent_total, numargs_total, &
                    sT_ts, pQ_ts, WaterBalance_ts, &
                    mT_ts, mQ_ts, mR_ts, C_Q_ts, ds_ts, dm_ts, dC_ts, SoluteBalance_ts)
   use cdf_gamma_mod
   use cdf_beta_mod
   !use cdf_normal_mod
   implicit none

   ! Start by declaring and initializing all the variables we will be using
   integer, intent(in) :: n_substeps, numflux, numsol, max_age, &
                          timeseries_length, numcomponent_total, numargs_total, len_index_ts
   real(8), intent(in) :: dt
   logical, intent(in) :: verbose, debug, warning, jacobian
   real(8), intent(in), dimension(0:timeseries_length - 1) :: J_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numflux - 1) :: Q_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numcomponent_total - 1) :: weights_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numargs_total - 1) :: SAS_args
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numargs_total - 1) :: P_list
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numsol - 1) :: C_J_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numflux - 1, 0:numsol - 1) :: alpha_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numsol - 1) :: k1_ts
   real(8), intent(in), dimension(0:timeseries_length - 1, 0:numsol - 1) :: C_eq_ts
   real(8), intent(in), dimension(0:numsol - 1) :: C_old
   real(8), intent(in), dimension(0:max_age - 1) :: sT_init_ts
   real(8), intent(in), dimension(0:max_age - 1, 0:numsol - 1) :: mT_init_ts
   integer, intent(in), dimension(0:len_index_ts - 1) :: index_ts
   integer, intent(in), dimension(0:numcomponent_total - 1) :: component_type
   integer, intent(in), dimension(0:numflux - 1) :: numcomponent_list
   integer, intent(in), dimension(0:numcomponent_total - 1) :: numargs_list
   real(8), intent(out), dimension(0:timeseries_length - 1, 0:numflux - 1, 0:numsol - 1) :: C_Q_ts
   real(8), intent(out), dimension(0:timeseries_length - 1, 0:numargs_total - 1, 0:numflux - 1, 0:numsol - 1) :: dC_ts
   real(8), intent(out), dimension(0:len_index_ts, 0:max_age - 1) :: sT_ts
   real(8), intent(out), dimension(0:len_index_ts, 0:numsol - 1, 0:max_age - 1) :: mT_ts
   real(8), intent(out), dimension(0:len_index_ts, 0:numargs_total - 1, 0:max_age - 1) :: ds_ts
   real(8), intent(out), dimension(0:len_index_ts, 0:numargs_total - 1, 0:numsol - 1, 0:max_age - 1) :: dm_ts
   real(8), intent(out), dimension(0:len_index_ts - 1, 0:numflux - 1, 0:max_age - 1) :: pQ_ts
   real(8), intent(out), dimension(0:len_index_ts - 1, 0:numflux - 1, 0:numsol - 1, 0:max_age - 1) :: mQ_ts
   real(8), intent(out), dimension(0:len_index_ts - 1, 0:numsol - 1, 0:max_age - 1) :: mR_ts
   real(8), intent(out), dimension(0:len_index_ts - 1, 0:max_age - 1) :: WaterBalance_ts
   real(8), intent(out), dimension(0:len_index_ts - 1, 0:numsol - 1, 0:max_age - 1) :: SoluteBalance_ts
   real(8), dimension(0:timeseries_length - 1, 0:numargs_total - 1, 0:numflux - 1) :: dW_ts
   real(8), dimension(0:timeseries_length - 1, 0:numflux - 1) :: P_old
   integer, dimension(0:numcomponent_total) :: args_index_list
   integer, dimension(0:numflux) :: component_index_list
   real(8), dimension(0:timeseries_length*n_substeps) :: STcum_top_start
   real(8), dimension(0:timeseries_length*n_substeps) :: STcum_bot_start
   real(8), dimension(0:timeseries_length*n_substeps - 1) :: STcum_bot
   real(8), dimension(0:timeseries_length*n_substeps - 1) :: STcum_top
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1) :: PQcum_bot
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1) :: PQcum_top
   integer, dimension(0:timeseries_length*n_substeps - 1, 0:numcomponent_total - 1) :: leftbreakpt_bot
   integer, dimension(0:timeseries_length*n_substeps - 1, 0:numcomponent_total - 1) :: leftbreakpt_top
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1) :: pQ_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1) :: pQ_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1, 0:numsol - 1) :: mQ_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numflux - 1, 0:numsol - 1) :: mQ_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numsol - 1) :: mR_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numsol - 1) :: mR_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1) :: fs_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1) :: fs_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numflux - 1) :: fsQ_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numflux - 1) :: fsQ_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: fm_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: fm_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numflux - 1, 0:numsol - 1) :: fmQ_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numflux - 1, 0:numsol - 1) :: fmQ_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: fmR_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: fmR_aver
   real(8), dimension(0:timeseries_length*n_substeps - 1) :: sT_start
   real(8), dimension(0:timeseries_length*n_substeps - 1) :: sT_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numsol - 1) :: mT_start
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numsol - 1) :: mT_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1) :: ds_start
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1) :: ds_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: dm_start
   real(8), dimension(0:timeseries_length*n_substeps - 1, 0:numargs_total - 1, 0:numsol - 1) :: dm_temp
   real(8), dimension(0:timeseries_length*n_substeps - 1) :: STcum_in
   integer, dimension(0:timeseries_length*n_substeps - 1) :: jt
   integer, dimension(0:timeseries_length*n_substeps - 1) :: jt_s
   real(8), dimension(0:timeseries_length - 1, 0:numargs_total - 1) :: grad
   integer :: iT_substep, iT, iT_s, iT_prev, jt_substep, jt_i
   real(8) :: one8, norm
   real(8) :: dS, dP, dSe, dPe, dSs, dPs
   real(8) :: h, hr
   real(8), dimension(4) :: rk_coeff
   real(8), dimension(5) :: rk_time
   character(len=128) :: tempdebugstring
   integer :: iq, s, M, N, ip, ic, c, rk, j_index_ts, jt_c
   integer :: carry
   integer :: leftbreakpt
   real(8) :: PQcum_component, X, scale_, loc_, a_arg, b_arg
   integer :: jt_this, topbot
   integer :: na
   integer :: ia
   integer :: i
   real(8) :: dif
   logical :: foundit
   real(8) :: start, finish

   C_Q_ts = 0.
   sT_ts = 0.
   mT_ts = 0.
   ds_ts = 0.
   dm_ts = 0.
   dC_ts = 0.
   dW_ts = 0.
   pQ_ts = 0.
   mQ_ts = 0.
   mR_ts = 0.
   WaterBalance_ts = 0.
   SoluteBalance_ts = 0.
   P_old = 1.
   args_index_list = 0
   component_index_list = 0
   STcum_top_start = 0.
   STcum_bot_start = 0.
   STcum_bot = 0.
   STcum_top = 0.
   PQcum_bot = 0.
   PQcum_top = 0.
   leftbreakpt_bot = 0
   leftbreakpt_top = 0
   pQ_temp = 0.
   pQ_aver = 0.
   mQ_temp = 0.
   mQ_aver = 0.
   mR_temp = 0.
   mR_aver = 0.
   fs_temp = 0.
   fs_aver = 0.
   fsQ_temp = 0.
   fsQ_aver = 0.
   fm_temp = 0.
   fm_aver = 0.
   fmQ_temp = 0.
   fmQ_aver = 0.
   fmR_temp = 0.
   fmR_aver = 0.
   sT_start = 0.
   sT_temp = 0.
   mT_start = 0.
   mT_temp = 0.
   ds_start = 0.
   ds_temp = 0.
   dm_start = 0.
   dm_temp = 0.
   iT_prev = -1

   call f_verbose('...Initializing arrays...')
   one8 = 1.0
   rk_time = (/0.0D0, 0.5D0, 0.5D0, 1.0D0, 1.0D0/)
   rk_coeff = (/1./6, 2./6, 2./6, 1./6/)
   norm = 1.0/n_substeps/n_substeps

   ! The list of probabilities in each sas function is a 1-D array.
   ! args_index_list gives the starting index of the probabilities (P) associated
   ! with each flux
   args_index_list(0) = 0
   component_index_list(0) = 0
   do iq = 0, numflux - 1
      component_index_list(iq + 1) = component_index_list(iq) + numcomponent_list(iq)
      do ic = component_index_list(iq), component_index_list(iq + 1) - 1
         args_index_list(ic + 1) = args_index_list(ic) + numargs_list(ic)
      end do
   end do
   call f_debug('args_index_list', one8*args_index_list(:))

   do iq = 0, numflux - 1
      do ic = component_index_list(iq), component_index_list(iq + 1) - 1
         if (component_type(ic) == -1) then
            do ia = 0, numargs_list(ic) - 1
               grad(:, args_index_list(ic) + ia) = &
               (P_list(:, args_index_list(ic) + ia + 1) - P_list(:, args_index_list(ic) + ia)) &
               /(SAS_args(:, args_index_list(ic) + ia + 1) - SAS_args(:, args_index_list(ic) + ia))
            end do
         end if
      end do
   end do
      
   ! modify the number of ages and the timestep by a facotr of n_substeps
   M = max_age*n_substeps
   N = timeseries_length*n_substeps
   h = dt/n_substeps

   call f_verbose('...Setting initial conditions...')
   sT_ts(0, :) = sT_init_ts
   mT_ts(0, :, :) = mT_init_ts

   call f_verbose('...Starting main loop...')
   ! Loop over ages
   do iT = 0, max_age - 1

      ! Start the substep loop
      do iT_substep = 0, n_substeps - 1

         iT_s = iT*n_substeps + iT_substep

         !!call cpu_time(start)
         do c = 0, N - 1
            jt_s(c) = mod(c + iT_s, N)
            jt_substep = mod(jt_s(c), n_substeps)
            ! jt(c) maps characteristic index c to the timestep it is currently intersecting
            jt(c) = (jt_s(c) - jt_substep)/n_substeps
         end do
         !!call cpu_time(finish)
         !!runtime(1) = runtime(1) + 1000*(finish-start)

         pQ_aver = 0
         mQ_aver = 0
         mR_aver = 0
         if (iT_s > 0) then
            sT_start(N - iT_s) = sT_init_ts(iT_prev)
            mT_start(N - iT_s, :) = mT_init_ts(iT_prev, :)
         end if

         sT_temp = sT_start
         mT_temp = mT_start

         if (jacobian) then
            fs_aver = 0
            fsQ_aver = 0
            fm_aver = 0
            fmQ_aver = 0
            fmR_aver = 0
            if (iT_s > 0) then
               ds_start(N - iT_s, :) = 0.
               dm_start(N - iT_s, :, :) = 0.
            end if
            ds_temp = ds_start
            dm_temp = dm_start
         end if

         ! This is the Runge-Kutta 4th order algorithm

         do rk = 1, 5
            hr = h*rk_time(rk)
            if (rk > 1) then

               ! ########################## vv NEW STATE vv ##########################
               ! Calculate the new age-ranked storage
               call f_debug('NEW STATE rk           ', (/rk*one8, iT_s*one8/))
               call f_debug('pQ_temp                ', pQ_temp(:, 0))
               call f_debug('sT_temp 0              ', sT_temp(:))
               sT_temp = sT_start ! Initial value
               mT_temp = mT_start + mR_temp*hr ! Initial value + reaction
               call f_debug('sT_temp 1              ', sT_temp(:))
               ! Fluxes in & out
               if (iT_s == 0) then
                  do concurrent(c=0:N - 1)
                     jt_c = jt(c)
                     sT_temp(c) = sT_temp(c) + J_ts(jt_c)*hr/h
                     mT_temp(c, :) = mT_temp(c, :) + J_ts(jt_c)*C_J_ts(jt_c, :)*(hr/h)
                  end do
                  call f_debug('sT_temp 2              ', sT_temp(:))
               end if
               do concurrent(c=0:N - 1)
                  jt_c = jt(c)
                  sT_temp(c) = sT_temp(c) - sum(Q_ts(jt_c, :)*pQ_temp(c, :))*hr
                  if (sT_temp(c) < 0) then
                     !call f_warning('WARNING: A value of sT is negative. Try increasing the number of substeps')
                     sT_temp(c) = 0
                  end if
               end do
               mT_temp = mT_temp - sum(mQ_temp, dim=2)*hr
               call f_debug('sT_temp 3              ', sT_temp(:))
               if (jacobian) then
                  ! Calculate new parameter sensitivity
                  ds_temp = ds_start - sum(fsQ_temp, dim=3)*hr
                  dm_temp = dm_start - sum(fmQ_temp, dim=3)*hr
                  ds_temp = ds_temp - fs_temp*hr
                  dm_temp = dm_temp - fm_temp*hr - fmR_temp*hr
               end if
               ! ########################## ^^ NEW STATE ^^ ##########################
            end if
            if (rk < 5) then
               call f_debug('GET FLUX  rk           ', (/rk*one8, iT_s*one8/))
               call f_debug('sT_temp                ', sT_temp(:))
               call f_debug('pQ_temp start          ', pQ_temp(:, 0))

               ! ########################## vv GET FLUX vv ##########################
               ! First get the cumulative age-ranked storage
               if ((iT_s == 0) .and. (hr == 0)) then
                  STcum_top = 0
                  PQcum_top = 0
                  leftbreakpt_top = -1
                  STcum_bot = 0
                  PQcum_bot = 0
                  leftbreakpt_bot = -1
                  pQ_temp = 0
               else
                  if (iT_s == 0) then
                     STcum_top = 0
                     STcum_bot = STcum_top + sT_temp*hr
                  else
                     do concurrent(c=0:N - 1)
                        STcum_top(c) = STcum_top_start(jt_s(c))*(1 - hr/h) + STcum_bot_start(jt_s(c) + 1)*(hr/h)
                     end do
                     STcum_bot = STcum_top + sT_temp*h
                  end if

                  PQcum_top = 0
                  PQcum_bot = 0
                  do topbot = 0, 1
                     ! Main lookup loop
                     ! This is where we calculate the SAS functions
                     if (topbot == 0) then
                        STcum_in = STcum_top
                     else
                        STcum_in = STcum_bot
                     end if
                     do concurrent(iq=0:numflux - 1)
                        do concurrent(ic=component_index_list(iq):component_index_list(iq + 1) - 1)
                           if (component_type(ic) == -1) then
                              do concurrent(c=0:N - 1)
                                 jt_c = jt(c)
                                 if (STcum_in(c) .le. SAS_args(jt_c, args_index_list(ic))) then
                                    if (topbot == 0) then
                                       PQcum_component = P_list(jt_c, args_index_list(ic))
                                       PQcum_top(c, iq) = PQcum_top(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                       leftbreakpt_top(c, ic) = -1
                                    else
                                       PQcum_component = P_list(jt_c, args_index_list(ic))
                                       PQcum_bot(c, iq) = PQcum_bot(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                       leftbreakpt_bot(c, ic) = -1
                                    end if
                                 else
                                    na = numargs_list(ic)
                                    ia = 0
                                    foundit = .FALSE.
                                    do i = 0, na - 1
                                       if (STcum_in(c) .lt. SAS_args(jt_c, args_index_list(ic) + i)) then
                                          ia = i - 1
                                          foundit = .TRUE.
                                          exit
                                       end if
                                    end do
                                    if (.not. foundit) then
                                       PQcum_component = P_list(jt_c, args_index_list(ic + 1) - 1)
                                       ia = na - 1
                                    else
                                       dif = STcum_in(c) - SAS_args(jt_c, args_index_list(ic) + ia)
                                       PQcum_component = P_list(jt_c, args_index_list(ic) + ia) &
                                       + dif*grad(jt_c, args_index_list(ic) + ia)
                                    end if
                                    if (topbot == 0) then
                                       PQcum_top(c, iq) = PQcum_top(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                       leftbreakpt_top(c, ic) = ia
                                    else
                                       PQcum_bot(c, iq) = PQcum_bot(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                       leftbreakpt_bot(c, ic) = ia
                                    end if
                                 end if
                              end do
                           elseif (component_type(ic) == 1) then
                              !Gamma distribution
                              do concurrent(c=0:N - 1, sT_temp(c) > 0)local (jt_c, loc_, scale_, a_arg, X, PQcum_component) shared (jt, SAS_args, STcum_in, PQcum_bot, PQcum_top, weights_ts)
                                 jt_c = jt(c)
                                 loc_ = SAS_args(jt_c, args_index_list(ic) + 0)
                                 scale_ = SAS_args(jt_c, args_index_list(ic) + 1)
                                 a_arg = SAS_args(jt_c, args_index_list(ic) + 2)
                                 X = (STcum_in(c) - loc_)/scale_
                                 PQcum_component = 0
                                 if (X .gt. 0) then
                                    PQcum_component = cum_gamma_fun(X, a_arg)
                                 end if
                                 if (topbot == 0) then
                                    PQcum_top(c, iq) = PQcum_top(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 else
                                    PQcum_bot(c, iq) = PQcum_bot(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 end if
                              end do
                           elseif (component_type(ic) == 2) then
                              !beta distribution
                              do concurrent(c=0:N - 1, sT_temp(c) > 0)
                                 jt_c = jt(c)
                                 loc_ = SAS_args(jt_c, args_index_list(ic) + 0)
                                 scale_ = SAS_args(jt_c, args_index_list(ic) + 1)
                                 a_arg = SAS_args(jt_c, args_index_list(ic) + 2)
                                 b_arg = SAS_args(jt_c, args_index_list(ic) + 3)
                                 X = (STcum_in(c) - loc_)/scale_
                                 X = MIN(MAX(0.0, X), 1.0)
                                 PQcum_component = 0
                                 if (X .gt. 0) then
                                    PQcum_component = cum_beta_fun(X, a_arg, b_arg)
                                 end if
                                 if (topbot == 0) then
                                    PQcum_top(c, iq) = PQcum_top(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 else
                                    PQcum_bot(c, iq) = PQcum_bot(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 end if
                              end do
                           elseif (component_type(ic) == 3) then
                              !kumaraswamy distribution
                              do concurrent(c=0:N - 1, sT_temp(c) > 0)local (jt_c, loc_, scale_, a_arg, b_arg, X, PQcum_component) shared (jt, SAS_args, STcum_in, PQcum_bot, PQcum_top, weights_ts)
                                 jt_c = jt(c)
                                 loc_ = SAS_args(jt_c, args_index_list(ic) + 0)
                                 scale_ = SAS_args(jt_c, args_index_list(ic) + 1)
                                 a_arg = SAS_args(jt_c, args_index_list(ic) + 2)
                                 b_arg = SAS_args(jt_c, args_index_list(ic) + 3)
                                 X = (STcum_in(c) - loc_)/scale_
                                 X = MIN(MAX(0.0, X), 1.0)
                                 PQcum_component = 1 - (1 - X**a_arg)**b_arg
                                 if (topbot == 0) then
                                    PQcum_top(c, iq) = PQcum_top(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 else
                                    PQcum_bot(c, iq) = PQcum_bot(c, iq) + weights_ts(jt_c, ic)*PQcum_component
                                 end if
                              end do
                           end if
                        end do
                     end do
                  end do

                  if (iT_s == 0) then
                     pQ_temp = (PQcum_bot - PQcum_top)/hr
                  else
                     pQ_temp = (PQcum_bot - PQcum_top)/h
                  end if
               end if

               do iq = 0, numflux - 1
                  where (sT_temp == 0)
                     pQ_temp(:, iq) = 0
                  end where
               end do

               ! Solute mass flux accounting
               mQ_temp = 0.
               do concurrent(s=0:numsol - 1, iq=0:numflux - 1, c=0:N - 1, sT_temp(c) .gt. 0)
                  ! Get the mass flux out
                  jt_c = jt(c)
                  mQ_temp(c, iq, s) = mT_temp(c, s)*alpha_ts(jt(c), iq, s)*Q_ts(jt(c), iq) &
                                      *pQ_temp(c, iq)/sT_temp(c)
               end do

               ! Reaction mass accounting
               ! If there are first-order reactions, get the total mass rate
               do concurrent(s=0:numsol - 1, c=0:N - 1, k1_ts(jt(c), s) .gt. 0)
                  jt_c = jt(c)
                  mR_temp(c, s) = k1_ts(jt_c, s)*(C_eq_ts(jt_c, s)*sT_temp(c) - mT_temp(c, s))
               end do

               if (jacobian) then
                  fs_temp = 0.
                  fsQ_temp = 0.
                  do concurrent(c=0:N - 1, iq=0:numflux - 1, sT_temp(c) .gt. 0)
                     fsQ_temp(c, :, iq) = fsQ_temp(c, :, iq) &
                                          + ds_temp(c, :)*pQ_temp(c, iq)*Q_ts(jt_c, iq)/sT_temp(c)
                  end do
                  do iq = 0, numflux - 1
                     do ic = component_index_list(iq), component_index_list(iq + 1) - 1
                        do c = 0, N - 1
                           jt_c = jt(c)
                           ! sensitivity to point before the start
                           if ((leftbreakpt_top(c, ic) >= 0) .and. (leftbreakpt_top(c, ic) < numargs_list(ic) - 1)) then
                              ip = args_index_list(ic) + leftbreakpt_top(c, ic)
                              call f_debug('iq, ic, ip, c ', (/iq*one8, ic*one8, ip*one8, c*one8/))
                              dS = SAS_args(jt_c, ip + 1) - SAS_args(jt_c, ip)
                              dP = P_list(jt_c, ip + 1) - P_list(jt_c, ip)
                              call f_debug('dP/dS start    ', (/dP/dS/))
                              fs_temp(c, ip) = fs_temp(c, ip) &
                                               + dP/(dS*dS)*sT_temp(c)*weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                           end if
                           ! sensitivity to point after the end
                           if ((leftbreakpt_bot(c, ic) + 1 > 0) .and. (leftbreakpt_bot(c, ic) + 1 <= numargs_list(ic) - 1)) then
                              ip = args_index_list(ic) + leftbreakpt_bot(c, ic) + 1
                              call f_debug('iq, ic, ip, c ', (/iq*one8, ic*one8, ip*one8, c*one8/))
                              dS = SAS_args(jt_c, ip) - SAS_args(jt_c, ip - 1)
                              dP = P_list(jt_c, ip) - P_list(jt_c, ip - 1)
                              call f_debug('dP/dS end      ', (/dP/dS/))
                              fs_temp(c, ip) = fs_temp(c, ip) &
                                               - dP/(dS*dS)*sT_temp(c)*weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                           end if
                           ! sensitivity to point within
                           if (leftbreakpt_bot(c, ic) > leftbreakpt_top(c, ic)) then
                              call f_debug('leftbreakpt_bot, _start', &
                                           (/leftbreakpt_bot(ic, c)*one8, leftbreakpt_top(ic, c)*one8/))
                              do leftbreakpt = leftbreakpt_top(c, ic) + 1, leftbreakpt_bot(c, ic)
                                 ip = args_index_list(ic) + leftbreakpt
                                 call f_debug('iq, ic, ip, c ', (/iq*one8, ic*one8, ip*one8, c*one8/))
                                 if (leftbreakpt > 0) then
                                    dSs = SAS_args(jt_c, ip) - SAS_args(jt_c, ip - 1)
                                    dPs = P_list(jt_c, ip) - P_list(jt_c, ip - 1)
                                 else
                                    dSs = 1.
                                    dPs = 0.
                                 end if
                                 if (leftbreakpt < numargs_list(ic) - 1) then
                                    dSe = SAS_args(jt_c, ip + 1) - SAS_args(jt_c, ip)
                                    dPe = P_list(jt_c, ip + 1) - P_list(jt_c, ip)
                                 else
                                    dSe = 1.
                                    dPe = 0.
                                 end if
                                 call f_debug('dP/dS middle   ', (/dPe/dSe, dPs/dSs/))
                                 fs_temp(c, ip) = fs_temp(c, ip) &
                                                  - (dPe/dSe - dPs/dSs)/h*weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                              end do
                           end if
                        end do
                     end do
                  end do
                  fm_temp = 0
                  fmQ_temp = 0
                  do s = 0, numsol - 1
                     do iq = 0, numflux - 1
                        do ip = 0, numargs_total - 1
                           do c = 0, N - 1
                              if (sT_temp(c) > 0) then
                                 fmQ_temp(c, ip, iq, s) = fmQ_temp(c, ip, iq, s) &
                                                          + dm_temp(c, ip, s)*alpha_ts(jt_c, iq, s)*Q_ts(jt_c, iq) &
                                                          *pQ_temp(c, iq)/sT_temp(c)
                              end if
                           end do
                        end do
                     end do
                  end do
                  do s = 0, numsol - 1
                     do ip = 0, numargs_total - 1
                        do c = 0, N - 1
                           jt_c = jt(c)
                           fmR_temp(c, ip, s) = fmR_temp(c, ip, s) &
                                                + k1_ts(jt_c, s)*(C_eq_ts(jt_c, s)*ds_temp(c, ip) - dm_temp(c, ip, s))
                        end do
                     end do
                  end do
                  do iq = 0, numflux - 1
                     do ic = component_index_list(iq), component_index_list(iq + 1) - 1
                        do c = 0, N - 1
                           jt_c = jt(c)
                           ! sensitivity to point before the start
                           if ((leftbreakpt_top(c, ic) >= 0) .and. (leftbreakpt_top(c, ic) < numargs_list(ic) - 1)) then
                              ip = args_index_list(ic) + leftbreakpt_top(c, ic)
                              dS = SAS_args(jt_c, ip + 1) - SAS_args(jt_c, ip)
                              dP = P_list(jt_c, ip + 1) - P_list(jt_c, ip)
                              fm_temp(c, ip, :) = fm_temp(c, ip, :) &
                                                  + dP/(dS*dS)*mT_temp(c, :) &
                                                  *alpha_ts(jt_c, iq, :)*weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                           end if
                           ! sensitivity to point after the end
                           if ((leftbreakpt_bot(c, ic) + 1 > 0) .and. (leftbreakpt_bot(c, ic) + 1 <= numargs_list(ic) - 1)) then
                              ip = args_index_list(ic) + leftbreakpt_bot(c, ic) + 1
                              dS = SAS_args(jt_c, ip) - SAS_args(jt_c, ip - 1)
                              dP = P_list(jt_c, ip) - P_list(jt_c, ip - 1)
                              fm_temp(c, ip, :) = fm_temp(c, ip, :) &
                                                  - dP/(dS*dS)*mT_temp(c, :) &
                                                  *alpha_ts(jt_c, iq, :)*weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                           end if
                           ! sensitivity to point within
                           if (leftbreakpt_bot(c, ic) > leftbreakpt_top(c, ic)) then
                              do leftbreakpt = leftbreakpt_top(c, ic) + 1, leftbreakpt_bot(c, ic)
                                 ip = args_index_list(ic) + leftbreakpt
                                 if (leftbreakpt > 0) then
                                    dSs = SAS_args(jt_c, ip) - SAS_args(jt_c, ip - 1)
                                    dPs = P_list(jt_c, ip) - P_list(jt_c, ip - 1)
                                 else
                                    dSs = 1.
                                    dPs = 0.
                                 end if
                                 if (leftbreakpt < numargs_list(ic) - 1) then
                                    dSe = SAS_args(jt_c, ip + 1) - SAS_args(jt_c, ip)
                                    dPe = P_list(jt_c, ip + 1) - P_list(jt_c, ip)
                                 else
                                    dSe = 1.
                                    dPe = 0.
                                 end if
                                 fm_temp(c, ip, :) = fm_temp(c, ip, :) &
                                                     - (dPe/dSe - dPs/dSs)*mT_temp(c, :)/sT_temp(c)/h &
                                                     *weights_ts(jt_c, ic)*Q_ts(jt_c, iq)
                              end do
                           end if
                        end do
                     end do
                  end do
               end if
               ! ########################## ^^ GET FLUX ^^ ##########################

               ! Average the rates of change using weights according to Runge-Kutta algorithm
               pQ_aver = pQ_aver + rk_coeff(rk)*pQ_temp
               mQ_aver = mQ_aver + rk_coeff(rk)*mQ_temp
               mR_aver = mR_aver + rk_coeff(rk)*mR_temp
               if (jacobian) then
                  fs_aver = fs_aver + rk_coeff(rk)*fs_temp
                  fsQ_aver = fsQ_aver + rk_coeff(rk)*fsQ_temp
                  fm_aver = fm_aver + rk_coeff(rk)*fm_temp
                  fmR_aver = fmR_aver + rk_coeff(rk)*fmR_temp
                  fmQ_aver = fmQ_aver + rk_coeff(rk)*fmQ_temp
               end if

               call f_debug('pQ_temp end            ', pQ_temp(:, 0))

            end if
            if (rk == 4) then
               call f_debug('FINALIZE  rk           ', (/rk*one8, iT_s*one8/))
               ! zero out the probabilities if there is no outflux this timestep
               do concurrent(iq=0:numflux - 1, c=0:N - 1, Q_ts(jt(c), iq) == 0)
                  pQ_aver(c, iq) = 0.
                  mQ_aver(c, iq, :) = 0.
               end do
               pQ_temp = pQ_aver
               mQ_temp = mQ_aver
               mR_temp = mR_aver
               if (jacobian) then
                  fs_temp = fs_aver
                  fsQ_temp = fsQ_aver
                  fm_temp = fm_aver
                  fmR_temp = fmR_aver
                  fmQ_temp = fmQ_aver
               end if
               call f_debug('pQ_aver                ', pQ_aver(:, 0))
            end if
         end do
         call f_debug_blank()

         ! Update the state with the new estimates
         sT_start = sT_temp
         mT_start = mT_temp
         if (jacobian) then
            ds_start = ds_temp
            dm_start = dm_temp
         end if
         ! Aggregate data from substep to timestep
         STcum_top_start(0) = STcum_bot_start(0)
         STcum_bot_start(0) = STcum_bot_start(0) + sT_init_ts(iT)*h
         do concurrent(c=0:N - 1)
            STcum_top_start(jt_s(c) + 1) = STcum_bot_start(jt_s(c) + 1)
            STcum_bot_start(jt_s(c) + 1) = STcum_bot_start(jt_s(c) + 1) + sT_start(c)*h
         end do

         ! update output conc and old water frac
         do concurrent(jt_i=0:timeseries_length - 1, jt_substep=0:n_substeps - 1, iq=0:numflux - 1, Q_ts(jt_i, iq) .gt. 0)
            c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
            C_Q_ts(jt_i, iq, :) = C_Q_ts(jt_i, iq, :) + mQ_aver(c, iq, :)*norm/Q_ts(jt_i, iq)*dt
         end do
         do concurrent(jt_i=0:timeseries_length - 1, jt_substep=0:n_substeps - 1, iq=0:numflux - 1, Q_ts(jt_i, iq) .gt. 0)
            c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
            P_old(jt_i, :) = P_old(jt_i, :) - pQ_aver(c, :)*norm*dt
         end do

         ! Get the timestep-averaged transit time distribution
         if (iT < max_age - 1) then
         do concurrent(j_index_ts=0:len_index_ts - 1, jt_substep=0:n_substeps - 1, jt_substep < iT_substep)
            jt_i = index_ts(j_index_ts)
            c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
            pQ_ts(j_index_ts, :, iT + 1) = pQ_ts(j_index_ts, :, iT + 1) + pQ_aver(c, :)*norm
            mQ_ts(j_index_ts, :, :, iT + 1) = mQ_ts(j_index_ts, :, :, iT + 1) + mQ_aver(c, :, :)*norm
            mR_ts(j_index_ts, :, iT + 1) = mR_ts(j_index_ts, :, iT + 1) + mR_aver(c, :)*norm
         end do
         end if

         do concurrent(j_index_ts=0:len_index_ts - 1, jt_substep=0:n_substeps - 1, jt_substep .ge. iT_substep)
            jt_i = index_ts(j_index_ts)
            c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
            pQ_ts(j_index_ts, :, iT) = pQ_ts(j_index_ts, :, iT) + pQ_aver(c, :)*norm
            mQ_ts(j_index_ts, :, :, iT) = mQ_ts(j_index_ts, :, :, iT) + mQ_aver(c, :, :)*norm
            mR_ts(j_index_ts, :, iT) = mR_ts(j_index_ts, :, iT) + mR_aver(c, :)*norm
         end do

         if (jacobian) then
            do j_index_ts = 0, len_index_ts - 1
               jt_i = index_ts(j_index_ts)
               do jt_substep = 0, n_substeps - 1
                  c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
                  do iq = 0, numflux - 1
                     if (Q_ts(jt_i, iq) > 0) then
                        dW_ts(jt_i, :, iq) = dW_ts(jt_i, :, iq) + fsQ_aver(c, :, iq)/Q_ts(jt_i, iq)*norm*dt
                        do ic = component_index_list(iq), component_index_list(iq + 1) - 1
                           do ip = args_index_list(ic), args_index_list(ic + 1) - 1
                              dW_ts(jt_i, ip, iq) = dW_ts(jt_i, ip, iq) + fs_aver(c, ip)/Q_ts(jt_i, iq)*norm*dt
                           end do
                        end do
                        dC_ts(jt_i, :, iq, :) = dC_ts(jt_i, :, iq, :) &
                                                + fmQ_aver(c, :, iq, :)/Q_ts(jt_i, iq)*norm*dt
                        do ic = component_index_list(iq), component_index_list(iq + 1) - 1
                           do ip = args_index_list(ic), args_index_list(ic + 1) - 1
                              dC_ts(jt_i, ip, iq, :) = dC_ts(jt_i, ip, iq, :) &
                                                       + fm_aver(c, ip, :)/Q_ts(jt_i, iq)*norm*dt
                           end do
                        end do
                     end if
                  end do
               end do
            end do
         end if

         do concurrent(j_index_ts=0:len_index_ts - 1)
            jt_i = index_ts(j_index_ts)
            ! Extract substep state at timesteps
            ! age-ranked storage at the end of the timestep
            jt_substep = n_substeps - 1
            c = mod(N + jt_i*n_substeps + jt_substep - iT_s, N)
            sT_ts(j_index_ts + 1, iT) = sT_ts(j_index_ts + 1, iT) + sT_start(c)/n_substeps
            ! parameter sensitivity
            ! Age-ranked solute mass
            mT_ts(j_index_ts + 1, :, iT) = mT_ts(j_index_ts + 1, :, iT) + mT_start(c, :)/n_substeps
            ! parameter sensitivity
            if (jacobian) then
               ds_ts(j_index_ts + 1, :, iT) = ds_ts(j_index_ts + 1, :, iT) + ds_start(c, :)/n_substeps
               dm_ts(j_index_ts + 1, :, :, iT) = dm_ts(j_index_ts + 1, :, :, iT) + dm_start(c, :, :)/n_substeps
            end if
         end do

         call f_debug('sT_ts(iT, :)     ', sT_ts(:, iT))
         call f_debug('pQ_ts(iT, :, 0)', pQ_ts(:, 0, iT))

         iT_prev = iT

      end do
      ! Print some updates
      if (mod(iT, 10) .eq. 0) then
         write (tempdebugstring, *) '...Done ', (iT), &
            'of', (max_age)
         call f_verbose(tempdebugstring)
      end if
   end do

   ! Calculate a water balance
   ! Difference of starting and ending age-ranked storage
   do iT = 0, max_age - 1
      do j_index_ts = 0, len_index_ts - 1
         jt_i = index_ts(j_index_ts)
         if (iT == 0) then
            WaterBalance_ts(j_index_ts, iT) = J_ts(jt_i) - sT_ts(j_index_ts + 1, iT)
         else
            WaterBalance_ts(j_index_ts, iT) = sT_ts(j_index_ts, iT - 1) - sT_ts(j_index_ts + 1, iT)
         end if
         ! subtract time-averaged water fluxes
         do iq = 0, numflux - 1
            WaterBalance_ts(j_index_ts, iT) = WaterBalance_ts(j_index_ts, iT) - &
                                              (Q_ts(j_index_ts, iq)*pQ_ts(j_index_ts, iq, iT))*dt
         end do

         ! Calculate a solute balance
         ! Difference of starting and ending age-ranked mass
         if (iT == 0) then
            do s = 0, numsol - 1
               SoluteBalance_ts(j_index_ts, s, iT) = C_J_ts(jt_i, s)*J_ts(jt_i) - mT_ts(j_index_ts + 1, s, iT)
            end do
         else
            SoluteBalance_ts(j_index_ts, :, iT) = mT_ts(j_index_ts, :, iT - 1) - mT_ts(j_index_ts + 1, :, iT)
         end if
         ! Subtract timestep-averaged mass fluxes
         do iq = 0, numflux - 1
            SoluteBalance_ts(j_index_ts, :, iT) = SoluteBalance_ts(j_index_ts, :, iT) - (mQ_ts(j_index_ts, iq, :, iT))*dt
         end do
         ! Reacted mass
         SoluteBalance_ts(j_index_ts, :, iT) = SoluteBalance_ts(j_index_ts, :, iT) + mR_ts(j_index_ts, :, iT)*dt
      end do

   end do ! End of main loop

   call f_verbose('...Finalizing...')

   ! From the old water concentration
   do concurrent(s=0:numsol - 1, iq=0:numflux - 1)
      where (Q_ts(:, iq) > 0)
         C_Q_ts(:, iq, s) = C_Q_ts(:, iq, s) + alpha_ts(:, iq, s)*C_old(s)*P_old(:, iq)
      end where
   end do

   if (jacobian) then
      do concurrent(s=0:numsol - 1, iq=0:numflux - 1, ip=0:numargs_total - 1)
         where (Q_ts(:, iq) > 0)
            dC_ts(:, ip, iq, s) = dC_ts(:, ip, iq, s) - C_old(s)*dW_ts(:, ip, iq)
         end where
      end do
   end if

   call f_verbose('...Finished...')

contains

   real(8) pure function cum_gamma_fun(X_, a_arg_)
      real(8), intent(in) :: X_, a_arg_
      cum_gamma_fun = cum_gamma_pure(X_, a_arg_)
   end function cum_gamma_fun

   real(8) pure function cum_beta_fun(X_, a_arg_, b_arg_)
      real(8), intent(in) :: X_, a_arg_, b_arg_
      cum_beta_fun = cum_beta_pure(X_, a_arg_, b_arg_)
   end function cum_beta_fun

   subroutine f_debug_blank()
      ! Prints a blank line
      if (debug) then
         print *, ''
      end if
   end subroutine f_debug_blank

   subroutine f_debug(debugstring, debugdblepr)
      ! Prints debugging information
      implicit none
      character(len=*), intent(in) :: debugstring
      real(8), dimension(:), intent(in) :: debugdblepr
      if (debug) then
         print 1, debugstring, debugdblepr
1        format(A26, *(f16.10))
      end if
   end subroutine f_debug

   subroutine f_warning(debugstring)
      ! Prints informative information
      implicit none
      !$acc routine seq
      character(len=*), intent(in) :: debugstring
      if (warning) then
         print *, debugstring
      end if
   end subroutine f_warning

   subroutine f_verbose(debugstring)
      ! Prints informative information
      implicit none
      !$acc routine seq
      character(len=*), intent(in) :: debugstring
      if (verbose) then
         print *, debugstring
      end if
   end subroutine f_verbose

end subroutine solveSAS
