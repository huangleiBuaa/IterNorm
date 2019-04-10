--[[
    The implementation of the Iterative Normalization for 4D data (inserted in Convolutional Neural Network), by using torch.bmm() for Group based whitening. This module has the same functionaliy as the nn.Spatial_DBN_PowerIter. It is more efficient, if the feature dimension/group number is large for Group based whitening.

    Author: Lei Huang
    Contact: lei.huang@inceptioniai.org
]]--
require('cunn')
local Spatial_DBN_PowerIter_BMM,parent = torch.class('nn.Spatial_DBN_PowerIter_BMM', 'nn.Module')

function Spatial_DBN_PowerIter_BMM:__init(nDim,m_perGroup, nIter,momentum,  affine)
    parent.__init(self)

  --the parameter 'affine' is used to scale the normalized output. If true, scale the output.
    if affine ~= nil then
        assert(type(affine) == 'boolean', 'affine has to be true/false')
        self.affine = affine
    else
        self.affine = false
    end

    if nIter ~= nil then
        self.nIter =nIter 
    else
        self.nIter = 5
    end

--the parameters 'm_perGroup' indicates the number in each group, which is used for group wised whitening
   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim 
   end 

    self.train = true
    self.debug = false
    self.GroupNumber=nDim/self.m_perGroup
    print('m_perGroup:'.. self.m_perGroup..'_GroupNumber:'..self.GroupNumber..'----nIter:'.. self.nIter)
    self.eps=1e-5
    self.nDim = nDim
    self.momentum = momentum or 0.1   --running average momentum

    self.eye_ngroup = torch.eye(self.m_perGroup)
    self.eye_ngroup_Block=torch.repeatTensor(self.eye_ngroup,self.GroupNumber,1,1)
    self.running_means=torch.Tensor(self.GroupNumber,self.m_perGroup):fill(0)  -- the mean used for inference, which is estimated based on each mini-batch with running average
    self.running_projections=self.eye_ngroup_Block:clone()  -- the whitening matrix used for inference, which is also estimated based on each mini-batch with running average

    if self.affine then
      print('-----------------using scaling-------------------')
        self.weight = torch.Tensor(nDim)
       self.bias = torch.Tensor(nDim)
       self.gradWeight = torch.Tensor(nDim)
       self.gradBias = torch.Tensor(nDim)
       self:reset()
    end

end

function Spatial_DBN_PowerIter_BMM:reset()
   self.weight:fill(1)
   self.bias:zero()
end

function Spatial_DBN_PowerIter_BMM:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
----------------------------- train mode---------------------
  function update_perGroup(data)
     local scale=data.new() --eigValue^(-1/2)
     local temp_1D= data.new()
     self.centered =self.centered or data.new()
     local output = data.new()
     self.sigma =self.sigma or data.new()
     
     local GroupNumber=data:size(1)
     local m_perGroup=data:size(2)
     local n_examples=data:size(3)
     
     self.centered:resizeAs(data)
     output:resizeAs(data)

     temp_1D:mean(data, 3)  -- E(x) = expectation of x.
     self.running_means:mul(1 - self.momentum):add(self.momentum, temp_1D)

      -- subtract mean
     self.centered:copy(data):add(-temp_1D:expandAs(self.centered))

      ----------------------calcualte the projection matrix----------------------
     self.sigma:resize(GroupNumber,m_perGroup,m_perGroup)
     self.sigma:bmm(self.centered,self.centered:transpose(2,3))
    -- sigma=sigma/n_examples
     self.sigma:div(n_examples):add(self.eps,self.eye_ngroup_Block)      
     self.trace=torch.cmul(self.sigma,self.eye_ngroup_Block):sum(2):sum(3)
     local sigma_norm=torch.cdiv(self.sigma,self.trace:expandAs(self.sigma))     
     local X=self.eye_ngroup_Block:clone() 
     for i=1,self.nIter do
        
       -- X=(3*X-X*X*X*sigma_norm)/2
        local tmp_X2=torch.bmm(X,X)
        local tmp_X3=torch.bmm(tmp_X2,X)
        local tmp_Xnorm=torch.bmm(tmp_X3,sigma_norm)
        X=(3*X-tmp_Xnorm)/2
        table.insert(self.set_X, X:clone())
      -- print(X)
     end
     
     self.whiten_matrix=torch.cdiv(X, torch.sqrt(self.trace):expandAs(X))
      self.running_projections:mul(1 - self.momentum):add(self.momentum, self.whiten_matrix) -- add to running projection
     output:bmm(self.whiten_matrix,self.centered)
      
      if self.debug then            
        print(self.running_means) 
        print(self.running_projections) 
      end
      ----------------record the results of per group--------------
      return output
 end
 
 
  ------------------------ test mode---------------------
 
  function test_perGroup(data)
    local GroupNumber = data:size(1)
    local m_perGroup=data:size(2)
    local n_examples=data:size(3)
    self.buffer_test=self.buffer_test or data.new()
    --local output=data.new()
    self.buffer_test:resizeAs(data):copy(data)
    self.buffer_test:add(-1,self.running_means:view(GroupNumber,m_perGroup,1):expandAs(data))

    self.buffer_test:bmm(self.running_projections,self.buffer_test)
    return self.buffer_test
  end
 
---------------------------------------------------------------------------------------
-------------------updateOutput main function-------------------------
----------------------------------------------------------------------------------------
   local nBatch = input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   self.set_X={}
    
   self.output=self.output or input.new()
   self.output:resize(#input)
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)

   self.gradInput=self.gradInput or input.new()
   self.gradInput:resize(#input)
  
   self.buffer = self.buffer or input.new()
   self.input_temp= self.input_temp or input.new() 
   self.output_temp= self.output_temp or input.new() 
   self.input_temp=input:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):reshape(self.GroupNumber,self.m_perGroup,nBatch*iH*iW)
   self.output_temp:resizeAs(self.input_temp)
   
   if self.train == false then
       self.output_temp=test_perGroup(self.input_temp)   
   else -- training mode
   -- print('-------------------------training mode-----------------') 
       self.output_temp=update_perGroup(self.input_temp)   
   end
     self.output:copy(self.output_temp:reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     self.normalized:copy(self.output)


     if self.affine then
        self.output:cmul(self.weight:view(1,nDim,1,1):expandAs(self.output))
        self.output:add(self.bias:view(1,nDim,1,1):expandAs(self.output))
     end
    collectgarbage()
    return self.output
end

function Spatial_DBN_PowerIter_BMM:updateGradInput(input, gradOutput)

   function Matrix_Pow3(input)
     local b=torch.bmm(input, input)
     return torch.bmm(b,input) 
   end

   function updateGradInput_perGroup(gradOutput_perGroup)
     local  sigma=self.sigma
     local  centered=self.centered
     local  whiten_matrix=self.whiten_matrix
     local  set_X=self.set_X
     local GroupNumber = gradOutput_perGroup:size(1) 
     local m_perGroup = gradOutput_perGroup:size(2) 
     local n_examples = gradOutput_perGroup:size(3) 
     local trace=self.trace
     --print(trace)
     local sigma_norm=torch.cdiv(sigma, trace:expandAs(sigma))
     self.dC=self.dC or gradOutput_perGroup.new()
     self.dA=self.dA or gradOutput_perGroup.new()
     self.dSigma=self.dSigma or gradOutput_perGroup.new()
     self.dXN=self.dXN or gradOutput_perGroup.new()
     self.f=self.f or gradOutput_perGroup.new()
     self.dC:resizeAs(whiten_matrix) 
     self.dA:resizeAs(whiten_matrix) 
     
     self.dC:bmm(gradOutput_perGroup, centered:transpose(2,3))
     --self.dXN=self.dC/torch.sqrt(trace) 
     self.dXN=torch.cdiv(self.dC, torch.sqrt(trace):expandAs(self.dC) )
--    print(#set_X)    
     local P3 
     if self.nIter==1 or self.nIter==0 then 
         P3=self.eye_ngroup_Block
     else
       P3=Matrix_Pow3(set_X[self.nIter-1])
     end
     --print(P3)
     self.dA:bmm(P3:transpose(2,3),self.dXN)
     local dX_kPlus=self.dXN
     for i=self.nIter-1, 1, -1 do
         ----calculate dL/dx_k+1--------------
       local X=set_X[i]
       local tmp_X2=torch.bmm(X,X)
       local tmp_Xsigma=torch.bmm(X,sigma_norm)

     
       --local tmp1=dX_kPlus*(X*X*sigma_norm):t()
       local tmp1_1=torch.bmm(tmp_X2, sigma_norm)
       local tmp1=torch.bmm(dX_kPlus, tmp1_1:transpose(2,3))

       --local tmp2=X:t()* dX_kPlus *(X*sigma_norm):t()
       local tmp2_1=torch.bmm(X:transpose(2,3),dX_kPlus)
       local tmp2=torch.bmm(tmp2_1,tmp_Xsigma:transpose(2,3))

       --local tmp3=(X*X):t()* dX_kPlus * sigma_norm:t() 
       local tmp3_1=torch.bmm(dX_kPlus, sigma_norm:transpose(2,3))
       local tmp3=torch.bmm(tmp_X2:transpose(2,3),tmp3_1)

      local dX_k= 3*dX_kPlus/2-(tmp1+tmp2+tmp3)/2 
       
      ------update dA-------------- 
       if  i ~= 1  then
         local    X_before=set_X[i-1]
         local tmp=Matrix_Pow3(X_before)
        -- self.dA:add(tmp:t()*dX_k)
        local tmp_dA=torch.bmm(tmp:transpose(2,3),dX_k)
        self.dA:add(tmp_dA)
           dX_kPlus=dX_k
       else 
         self.dA:add(dX_k) 
       end
     end
    
     self.dA=-self.dA/2

      local tmp_trace32=torch.pow(trace, 3/2)*2
     if self.nIter==0 then
    --self.dSigma=-torch.trace(self.dC)/(2*trace^(3/2)) * eye_nDim
      local tmp_1=torch.cmul(self.dC, self.eye_ngroup_Block):sum(2):sum(3)
      local tmp_2= -torch.cdiv(tmp_1,tmp_trace32)
      self.dSigma= torch.cmul(self.eye_ngroup_Block, tmp_2:expandAs(self.eye_ngroup_Block))
     else
        --local s1=torch.trace(self.dA:t()*sigma)
        --local s2=torch.trace(self.dC:t()*set_X[self.nIter]) 
        local s1=torch.cmul(self.dA,sigma):sum(2):sum(3)
        local s2=torch.cmul(self.dC,set_X[self.nIter]):sum(2):sum(3)
        --self.dSigma=-(s1/(trace^2)+s2/(2*trace^(3/2)))*eye_nDim
        --self.dSigma=self.dSigma+self.dA/trace
        local tmp_1=torch.cdiv(s1, torch.pow(trace,2))
        local tmp_2=torch.cdiv(s2, tmp_trace32)
        local tmp_3=-(tmp_1+tmp_2)
        self.dSigma=torch.cmul(self.eye_ngroup_Block, tmp_3:expandAs(self.eye_ngroup_Block))
        local tmp_4=torch.cdiv(self.dA, trace:expandAs(self.dA))
        self.dSigma=self.dSigma + tmp_4
     end
     
     local dSigma_sym=(self.dSigma+self.dSigma:transpose(2,3))/2

     self.f:mean(gradOutput_perGroup, 3)
     local d_mean=gradOutput_perGroup-self.f:expandAs(gradOutput_perGroup) 
     self.buffer:resizeAs(gradOutput_perGroup) 
     self.buffer:bmm(whiten_matrix,d_mean) 
     --local gradInput=(2/nBatch)*centered*dSigma_sym
     local gradInput=torch.bmm(dSigma_sym,centered)
     gradInput=gradInput * (2/n_examples)
     gradInput=gradInput+self.buffer  
    
     return gradInput
  end
  

----------------------------------------------------------------  
 -------update the gradInput per Group in test mode:  this mode may be benifit based on the Batch Renomrlaizaiton methods----------------------------------------------------- 
  function updateGradInput_perGroup_test(gradOutput_perGroup)
     local running_projection=self.running_projections --use running projection
     local gradInput=gradOutput_perGroup.new()
     gradInput:resizeAs(gradOutput_perGroup)
     gradInput:bmm(running_projection,gradOutput_perGroup)
     return gradInput
  end
---------------------------------------------------------------------------------------
-------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------


   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   local nBatch=input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.affine then
     self.gradInput:cmul(self.weight:view(1,nDim,1,1):expandAs(self.gradInput))
   end

   self.input_temp=self.gradInput:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(self.GroupNumber,self.m_perGroup,nBatch*iH*iW)
   self.output_temp:resizeAs(self.input_temp)
  
   if self.train==false then 
         self.output_temp=updateGradInput_perGroup_test(self.input_temp)   
   else
         self.output_temp=updateGradInput_perGroup(self.input_temp)   
   end 
   self.gradInput:copy(self.output_temp:reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     
   collectgarbage()
   return self.gradInput
end

function Spatial_DBN_PowerIter_BMM:setTrainMode(isTrain)
  if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
  else
    self.train=true  

  end
end

function Spatial_DBN_PowerIter_BMM:accGradParameters(input, gradOutput, scale)
  if self.affine then
    scale = scale or 1.0
    local nBatch = input:size(1)
    local nFeature = input:size(2)
    local iH = input:size(3)
    local iW = input:size(4)
    self.output_temp:resizeAs(self.normalized):copy(self.normalized)
    self.output_temp = self.output_temp:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
    self.buffer:sum(self.output_temp, 1) -- sum over mini-batch
    self.output_temp:sum(self.buffer, 3) -- sum over pixels
    self.gradWeight:add(scale, self.output_temp)
    self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
    self.output_temp:sum(self.buffer, 3)
    self.gradBias:add(scale, self.output_temp) -- sum over mini-batch
  end
end
function Spatial_DBN_PowerIter_BMM:clearState()
  print('-----------clear the memory--------------------')
       -- self.buffer:set()
       -- self.output_temp:set()
      --   self.input_temp:set()
      --   self.gradInput:set()
      --   self.dC:set()
      --   self.dSigma:set()
      --   self.dXN:set()
      --   self.f:set()
      --  self.eye_ngroup:set()
      --  self.eye_ngroup_last:set()
      --  self.output:set()
  return 
end
