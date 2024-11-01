// Profile.jsx
import React from 'react';
import { Box, Grid } from "@chakra-ui/react";
import ProfileInfoCard from "views/admin/dataTables/components/ProfileInfoCard";
import TokenBalanceCard from "views/admin/dataTables/components/TokenBalanceCard";
import { useSelector } from 'react-redux';

// Define the token contract address for $FML
const tokenAddress = "0xc8d94c5cB4462966473b3b1505B8129f12152977"; // Replace with actual token contract address

export default function Profile() {
    const storedName = useSelector((state) => state.person.name);
    const storedAddress = useSelector((state) => state.person.address);

    const userData = {
        name: storedName,
        walletId: storedAddress,
    };

    return (
        <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
            <Grid
                templateColumns={{
                    base: "1fr",
                    md: "1fr 1fr",
                }}
                gap="20px"
            >
                {/* Profile Information Card */}
                <ProfileInfoCard
                    name={userData.name}
                    walletId={userData.walletId}
                />

                {/* Token Balance Card */}
                <TokenBalanceCard walletId={storedAddress} tokenAddress={tokenAddress} />
            </Grid>
        </Box>
    );
}
