#include<bits/stdc++.h>
int read(){ int x; scanf("%d",&x); return x; }
int main()
{
	int n=read(),m=read();
	static int a[30][30];
	int ans=0,Ans=0;
	for(int i=0;i<m;i++) a[read()][read()]++;
	for(int i=0;i<(1<<(n-1));i++)
	{
		int _ans=0;
		for(int j=0;j<n;j++) for(int k=0;k<n;k++) if(a[j][k]&&((i>>j)&1)!=((i>>k)&1)) _ans++;
		if(_ans>ans){ ans=_ans; Ans=i; }
	}
	printf("%d\n",ans);
	for(int i=0;i<n;i++) printf("%d%c",(Ans>>i)&1,i==n-1?'\n':' ');
	return 0;
}
