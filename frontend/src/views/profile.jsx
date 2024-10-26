// Profile.jsx

import React from 'react';
import { Box, Grid } from "@chakra-ui/react";
import ProfileInfoCard from "views/admin/dataTables/components/ProfileInfoCard";
import TokenBalanceCard from "views/admin/dataTables/components/TokenBalanceCard";

// Placeholder for user data
const userData = {
  name: "Adela Parkson",
  walletId: "0x1234...abcd",
  tokens: 1250,
  avatarSrc: "assets/img/avatars/avatar4.png"  // Use a path to the avatar image
};

export default function Profile() {
  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
      <Grid
        templateColumns={{
          base: "1fr",
          md: "1fr 1fr", // Split into two columns on medium screens and larger
        }}
        gap="20px"
      >
        {/* Profile Information Card */}
        <ProfileInfoCard
          name={userData.name}
          walletId={userData.walletId}
          avatarSrc={userData.avatarSrc}
        />

        {/* Token Balance Card */}
        <TokenBalanceCard tokens={userData.tokens} />
      </Grid>
    </Box>
  );
}
