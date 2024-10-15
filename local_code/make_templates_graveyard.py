import make_templates

def seq_diff(A,B):
    N,n,_,d=A.shape
    res_tensor_list=[]
    for state_index in range(N):
        state_mat=A[state_index] #nxd state

        #Now need to find all the differences
        
        state_mat_expanded = state_mat.unsqueeze(0).expand(N, -1, -1)  # Nxmxd tensor

        # Compute the differences
        differences = A.unsqueeze(2) - state_mat_expanded.unsqueeze(1)  # Nxmxmx d tensor

        # Compute the absolute differences and sum along the last dimension
        abs_differences = torch.abs(differences)  # Nxmxmx d tensor
        sum_abs_differences = abs_differences.sum(dim=3)  # Nxmxm tensor
        res_tensor_list.append(sum_abs_differences)
    res_tensor_list=torch.stack(res_tensor_list,dim=0)

    return res_tensor_list
        






def seq_diff2(A,B):
    N,n,d=A.shape
    res_tensor_list=[]
    for state_index in range(N):
        state_mat=A[state_index] #nxd state
        state_matches_list=[]
        for particle_index in range(n):
            B_comp=B[:,particle_index,:,:] #Nxd state
            print(f'b comp')
            print(B_comp)
            same_rows = (B_comp.unsqueeze(1) == B_comp.unsqueeze(0)).all(-1)
            row_pairs = torch.nonzero(same_rows.triu(1), as_tuple=True)
            print(row_pairs)
            if row_pairs[0].shape[0] != 0:
                continue
            else:
                print(f'the else')
                #Now need to find all the differences
                
                # B_comp = B_comp.unsqueeze(1)  # Nx1xd tensor
                # B_comp = B_comp.repeat(1,n,1,1)  # Nxmxd tensor
                B_comp_1 = B_comp.unsqueeze(1)  # Shape becomes (N, 1, m, d)
                state_mat=state_mat.unsqueeze(0).unsqueeze(2)
                row_diffs=B_comp_1-state_mat
                row_diffs=row_diffs.transpose(1,2)
                print(row_diffs.shape)
                exit()
                print(row_diffs[0])
                
                print(B_comp_1.shape)
                
                print(f'initial state')
                print(B_comp_1[0][0])
                print(B_comp_1[1][0])
                print(f'state mat subtracting off')
                print(state_mat[0])
                print(f'difference')
                print(row_diffs[0])
                print(f'sum of abs diff:')
                print(torch.abs(row_diffs[0]).sum(dim=2))
                exit()
                # B_comp_2 = B_comp.unsqueeze(2)  # Shape becomes (N, m, 1, d)
                
                # # Step 2: Perform outer product
                # print(B_comp.shape)
                # exit()
                # B_comp = B_comp_1 - B_comp_2  # Broadcasting creates (N, m, m, d)
                # print(B_comp[0].shape)
                # print(B_comp[0])
                # print(state_mat)
                
                # exit()
                
                # Compute the differences
                #differences = B_comp-state_mat  # Nxmxmxd tensor
                
                # Compute the absolute differences and sum along the last dimension
                abs_differences = torch.abs(row_diffs)  # Nxmxmx d tensor
                print(abs_differences.shape)
                
                sum_abs_differences = abs_differences.sum(dim=3)  # Nxmxm tensor
                print(sum_abs_differences.shape)
                
                #Instead of keeping the whole thing, at this point you may as well identify which of the N indices you map to return an n tensor

                
                zeros=count_zeros(sum_abs_differences)
                print(zeros[0])
                exit()
                print(zeros[0][0])

                print(sum_abs_differences[zeros[0]])
                print(state_mat)
                print(B_comp[zeros[0]][2])

                exit()
                print(f'len zeros {len(zeros)}')
                print(f'A first entry {A(zeros[0][0])}')
                exit()
                
                
                state_matches_list.append(sum_abs_differences)
                #state_matches_tensor=torch.stack(state_matches_list,dim=1)
    
        res_tensor_list.append(state_matches_list)
    #res_tensor_list=torch.stack(res_tensor_list,dim=0)

            
            

    return res_tensor_list