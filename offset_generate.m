function[offset_store_level]= offset_generate(level,num_node,id_l,child_num)
for l=1:level-1
     n_node=num_node{l};
     id_level=id_l{l};
     offset_temp=zeros(1,n_node);
     for k=1:n_node
         if(id_level(k)==0)
         if (k==1)
          offset_temp(k)=0;
         elseif(id_level(k-1)==1)
         offset_temp(k)=offset_temp(k-1);
         else
           
             offset_temp(k)=child_num{l}(k-1)+offset_temp(k-1);
         end
         else
             if(k==1)
                 offset_temp(k)=0;
             else
             offset_temp(k)=offset_temp(k-1)+child_num{l}(k-1);
             end
         end
         
         
     end
     offset_store_level{l}=offset_temp;
end 
end


