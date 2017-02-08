regsubsets(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
             avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif +
             wrat2 + wrat3 + hinc2 + hinc3 + ln_chld + sr_chld + ln_avhv + sr_avhv +
             ln_incm + sr_incm + ln_inca + sr_inca + ln_plow + sr_plow + ln_npro + sr_npro +
             ln_tgif + sr_tgif + ln_lgif + sr_lgif + ln_rgif + sr_rgif + ln_tdon + sr_tdon +
             ln_tlag + sr_tlag + ln_agif + sr_agif, data.train.std.c, nvmax = 24)

# M2d: my best selection of variables
# model.lda2 = lda(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + hinc3 + wrat + 
#                    wrat2 + wrat3 + ln_incm + ln_npro + ln_plow + ln_tgif + ln_lgif + 
#                    ln_agif + ln_tdon + ln_tlag + sr_plow + sr_tdon + sr_tlag, data.train.std.c)