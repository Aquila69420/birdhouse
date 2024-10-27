// Profile.jsx

import React from 'react';
import { Box, Grid } from "@chakra-ui/react";
import ProfileInfoCard from "views/admin/dataTables/components/ProfileInfoCard";
import TokenBalanceCard from "views/admin/dataTables/components/TokenBalanceCard";
import { useDispatch, useSelector } from 'react-redux';

// Placeholder for user data

export default function Profile() {
  const storedName = useSelector((state) => state.person.name);
  const storedAddress = useSelector((state) => state.person.address);
  
  const userData = { 
    name: storedName,
    walletId: storedAddress,
    tokens: 1250, // Use a path to the avatar image
  };
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
